import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

class CloudDataManager:
    def __init__(self, app_id: str, app_secret: str, env_id: str, workspace_root: str = "workspace"):
        self.app_id = app_id
        self.app_secret = app_secret
        self.env_id = env_id
        self.workspace_root = Path(workspace_root)
        self.access_token = None

    def _get_access_token(self) -> str:
        """获取 access_token（复用原函数）"""
        url = "https://api.weixin.qq.com/cgi-bin/token"
        params = {
            "grant_type": "client_credential",
            "appid": self.app_id,
            "secret": self.app_secret
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if "access_token" not in data:
            raise RuntimeError(f"获取 access_token 失败: {data}")
        self.access_token = data["access_token"]
        return self.access_token

    def fetch_patient_names(self) -> List[str]:
        """从云端获取所有患者名称，去重排序后返回"""
        if not self.access_token:
            self._get_access_token()

        url = f"https://api.weixin.qq.com/tcb/databasequery?access_token={self.access_token}"
        all_records = []
        skip = 0
        limit = 100
        while True:
            query = f'''
            db.collection("cases")
              .skip({skip})
              .limit({limit})
              .get()
            '''
            payload = {"env": self.env_id, "query": query}
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("errcode", 0) != 0:
                raise RuntimeError(f"查询失败: {data}")

            raw_list = data.get("data", [])
            if not raw_list:
                break
            for item in raw_list:
                try:
                    all_records.append(json.loads(item))
                except Exception:
                    continue
            if len(raw_list) < limit:
                break
            skip += limit

        # 提取 patientName
        patients = set()
        for rec in all_records:
            name = rec.get("patientName")
            if name:
                patients.add(name)
        return sorted(patients)

    def download_patients(self, patient_names: List[str], progress_callback: Optional[Callable] = None) -> tuple:
        mask_files = []
        overlay_map = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for name in patient_names:
                future = executor.submit(self._download_patient, name, progress_callback)
                futures.append(future)

            for future in as_completed(futures):
                mf, om = future.result()
                mask_files.extend(mf)
                overlay_map.update(om)

        return mask_files, overlay_map

    def _download_patient(self, name: str, progress_callback: Optional[Callable]) -> tuple:
        """下载单个患者的所有文件"""
        patient_dir = self.workspace_root / name
        mask_dir = patient_dir / "mask"
        overlay_dir = patient_dir / "overlay"
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        records = self._query_patient_records(name)
        file_items = self._collect_file_items(records)

        file_ids = [item["fileid"] for item in file_items]
        url_items = self._get_download_urls(file_ids)
        url_map = {item["fileid"]: item for item in url_items}

        mask_files = []
        for item in file_items:
            fid = item["fileid"]
            file_type = item["type"]
            url_info = url_map.get(fid)
            if not url_info or url_info.get("status", -1) != 0:
                continue
            download_url = url_info.get("download_url")
            if not download_url:
                continue

            filename = fid.split("/")[-1]
            if file_type == "mask":
                save_path = mask_dir / filename
                mask_files.append(str(save_path))
            elif file_type == "overlay":
                save_path = overlay_dir / filename
            else:
                save_dir = patient_dir / file_type
                save_dir.mkdir(exist_ok=True)
                save_path = save_dir / filename

            self._download_file(download_url, save_path)
            if progress_callback:
                progress_callback(f"已下载: {name}/{file_type}/{filename}")

        # 构建 overlay 映射
        overlay_map = {}
        for mask_path in mask_dir.glob("*_mask.png"):
            mask_filename = mask_path.name
            overlay_filename = mask_filename.replace("_mask", "_overlay")
            overlay_path = overlay_dir / overlay_filename
            if overlay_path.exists():
                overlay_map[mask_filename] = str(overlay_path)

        return mask_files, overlay_map

    def _query_patient_records(self, patient_name: str) -> List[dict]:
        """查询单个患者的记录（复用原 query_cases）"""
        if not self.access_token:
            self._get_access_token()
        url = f"https://api.weixin.qq.com/tcb/databasequery?access_token={self.access_token}"
        query = f'''
        db.collection("cases")
          .where({{
            patientName: "{patient_name}"
          }})
          .limit(100)
          .get()
        '''
        payload = {"env": self.env_id, "query": query}
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("errcode", 0) != 0:
            raise RuntimeError(f"查询失败: {data}")
        raw_list = data.get("data", [])
        records = []
        for item in raw_list:
            try:
                records.append(json.loads(item))
            except Exception:
                continue
        return records

    def _collect_file_items(self, records: List[dict]) -> List[dict]:
        """从记录中提取文件项（复用原 collect_file_ids）"""
        items = []
        for r in records:
            patient_name = r.get("patientName", "unknown")
            record_id = r.get("_id", "")
            for key in ["imageFileID", "maskFileID", "overlayFileID"]:
                fid = r.get(key)
                if fid:
                    items.append({
                        "type": key.replace("FileID", "").lower(),
                        "patientName": patient_name,
                        "recordId": record_id,
                        "fileid": fid
                    })
        return items

    def _get_download_urls(self, file_ids: List[str]) -> List[dict]:
        """批量获取下载链接（复用原 get_download_urls）"""
        url = f"https://api.weixin.qq.com/tcb/batchdownloadfile?access_token={self.access_token}"
        payload = {
            "env": self.env_id,
            "file_list": [{"fileid": fid, "max_age": 3600} for fid in file_ids]
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("errcode", 0) != 0:
            raise RuntimeError(f"获取下载链接失败: {data}")
        return data.get("file_list", [])

    def _download_file(self, url: str, save_path: Path):
        """下载单个文件（复用原 download_file）"""
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)