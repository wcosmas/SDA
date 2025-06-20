"""
Part B: Storage Manager Component
RTV Senior Data Scientist Technical Assessment

This module handles secure data storage across multiple platforms:
- AWS S3 storage
- Azure Blob Storage
- Local file system storage
- Database storage
- Data encryption and security
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import boto3
from azure.storage.blob import BlobServiceClient
import structlog
from cryptography.fernet import Fernet
import hashlib

from config.pipeline_config import config

logger = structlog.get_logger(__name__)


class StorageManager:
    """Unified storage manager for multiple storage backends"""
    
    def __init__(self):
        self.config = config
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Initialize storage clients based on configuration
        self.storage_config = self.config.get_storage_config()
        self._initialize_storage_clients()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive data"""
        key_file = Path("config/encryption.key")
        
        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            key_file.parent.mkdir(exist_ok=True)
            with open(key_file, "wb") as f:
                f.write(key)
            logger.info("Generated new encryption key")
            return key
    
    def _initialize_storage_clients(self):
        """Initialize storage clients based on configuration"""
        self.s3_client = None
        self.azure_client = None
        
        if self.storage_config["provider"] == "aws":
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.storage_config["access_key"],
                    aws_secret_access_key=self.storage_config["secret_key"],
                    region_name=self.storage_config["region"]
                )
                logger.info("Initialized AWS S3 client")
            except Exception as e:
                logger.error("Failed to initialize S3 client", error=str(e))
        
        elif self.storage_config["provider"] == "azure":
            try:
                self.azure_client = BlobServiceClient(
                    account_url=f"https://{self.storage_config['account_name']}.blob.core.windows.net",
                    credential=self.storage_config["account_key"]
                )
                logger.info("Initialized Azure Blob client")
            except Exception as e:
                logger.error("Failed to initialize Azure client", error=str(e))
    
    async def store_data(self, data: pd.DataFrame, path: str, 
                        encrypt_sensitive: bool = True) -> Dict[str, Any]:
        """Store data in the configured storage backend"""
        try:
            # Generate metadata
            metadata = self._generate_storage_metadata(data, path)
            
            # Encrypt sensitive columns if required
            if encrypt_sensitive:
                data = self._encrypt_sensitive_data(data.copy())
            
            # Convert to appropriate format
            file_format = Path(path).suffix.lower()
            if file_format in ['.parquet', '']:
                file_format = '.parquet'  # Default to parquet
                if not path.endswith('.parquet'):
                    path += '.parquet'
            
            # Store based on configured provider
            if self.storage_config["provider"] == "aws":
                result = await self._store_to_s3(data, path, metadata)
            elif self.storage_config["provider"] == "azure":
                result = await self._store_to_azure(data, path, metadata)
            else:
                result = await self._store_locally(data, path, metadata)
            
            logger.info("Data stored successfully", path=path, records=len(data))
            return result
            
        except Exception as e:
            logger.error("Data storage failed", path=path, error=str(e))
            raise
    
    async def retrieve_data(self, path: str, decrypt_sensitive: bool = True) -> pd.DataFrame:
        """Retrieve data from storage"""
        try:
            # Retrieve based on configured provider
            if self.storage_config["provider"] == "aws":
                data = await self._retrieve_from_s3(path)
            elif self.storage_config["provider"] == "azure":
                data = await self._retrieve_from_azure(path)
            else:
                data = await self._retrieve_locally(path)
            
            # Decrypt sensitive columns if required
            if decrypt_sensitive:
                data = self._decrypt_sensitive_data(data)
            
            logger.info("Data retrieved successfully", path=path, records=len(data))
            return data
            
        except Exception as e:
            logger.error("Data retrieval failed", path=path, error=str(e))
            raise
    
    async def list_files(self, prefix: str = "") -> List[Dict]:
        """List files in storage with metadata"""
        try:
            if self.storage_config["provider"] == "aws":
                files = await self._list_s3_files(prefix)
            elif self.storage_config["provider"] == "azure":
                files = await self._list_azure_files(prefix)
            else:
                files = await self._list_local_files(prefix)
            
            return files
            
        except Exception as e:
            logger.error("File listing failed", prefix=prefix, error=str(e))
            raise
    
    async def delete_data(self, path: str) -> bool:
        """Delete data from storage"""
        try:
            if self.storage_config["provider"] == "aws":
                success = await self._delete_from_s3(path)
            elif self.storage_config["provider"] == "azure":
                success = await self._delete_from_azure(path)
            else:
                success = await self._delete_locally(path)
            
            if success:
                logger.info("Data deleted successfully", path=path)
            return success
            
        except Exception as e:
            logger.error("Data deletion failed", path=path, error=str(e))
            raise
    
    async def _store_to_s3(self, data: pd.DataFrame, path: str, metadata: Dict) -> Dict:
        """Store data to AWS S3"""
        if not self.s3_client:
            raise ValueError("S3 client not initialized")
        
        # Convert DataFrame to bytes
        buffer = self._dataframe_to_bytes(data, Path(path).suffix)
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.storage_config["bucket"],
            Key=path,
            Body=buffer,
            Metadata={k: str(v) for k, v in metadata.items()}
        )
        
        return {
            "provider": "aws",
            "bucket": self.storage_config["bucket"],
            "key": path,
            "size": len(buffer),
            "metadata": metadata
        }
    
    async def _store_to_azure(self, data: pd.DataFrame, path: str, metadata: Dict) -> Dict:
        """Store data to Azure Blob Storage"""
        if not self.azure_client:
            raise ValueError("Azure client not initialized")
        
        # Convert DataFrame to bytes
        buffer = self._dataframe_to_bytes(data, Path(path).suffix)
        
        # Upload to Azure
        blob_client = self.azure_client.get_blob_client(
            container=self.storage_config["container"],
            blob=path
        )
        
        blob_client.upload_blob(
            buffer, 
            overwrite=True,
            metadata=metadata
        )
        
        return {
            "provider": "azure",
            "container": self.storage_config["container"],
            "blob": path,
            "size": len(buffer),
            "metadata": metadata
        }
    
    async def _store_locally(self, data: pd.DataFrame, path: str, metadata: Dict) -> Dict:
        """Store data locally"""
        local_path = Path(self.storage_config["path"]) / path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        file_format = Path(path).suffix.lower()
        if file_format == '.parquet':
            data.to_parquet(local_path, index=False)
        elif file_format == '.csv':
            data.to_csv(local_path, index=False)
        elif file_format == '.json':
            data.to_json(local_path, orient='records')
        else:
            data.to_parquet(local_path, index=False)  # Default to parquet
        
        # Save metadata
        metadata_path = local_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return {
            "provider": "local",
            "path": str(local_path),
            "size": local_path.stat().st_size,
            "metadata": metadata
        }
    
    async def _retrieve_from_s3(self, path: str) -> pd.DataFrame:
        """Retrieve data from AWS S3"""
        if not self.s3_client:
            raise ValueError("S3 client not initialized")
        
        response = self.s3_client.get_object(
            Bucket=self.storage_config["bucket"],
            Key=path
        )
        
        return self._bytes_to_dataframe(response['Body'].read(), Path(path).suffix)
    
    async def _retrieve_from_azure(self, path: str) -> pd.DataFrame:
        """Retrieve data from Azure Blob Storage"""
        if not self.azure_client:
            raise ValueError("Azure client not initialized")
        
        blob_client = self.azure_client.get_blob_client(
            container=self.storage_config["container"],
            blob=path
        )
        
        data = blob_client.download_blob().readall()
        return self._bytes_to_dataframe(data, Path(path).suffix)
    
    async def _retrieve_locally(self, path: str) -> pd.DataFrame:
        """Retrieve data from local storage"""
        local_path = Path(self.storage_config["path"]) / path
        
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        
        file_format = Path(path).suffix.lower()
        if file_format == '.parquet':
            return pd.read_parquet(local_path)
        elif file_format == '.csv':
            return pd.read_csv(local_path)
        elif file_format == '.json':
            return pd.read_json(local_path)
        else:
            return pd.read_parquet(local_path)  # Default to parquet
    
    async def _list_s3_files(self, prefix: str) -> List[Dict]:
        """List files in S3"""
        if not self.s3_client:
            return []
        
        response = self.s3_client.list_objects_v2(
            Bucket=self.storage_config["bucket"],
            Prefix=prefix
        )
        
        files = []
        for obj in response.get('Contents', []):
            files.append({
                "path": obj['Key'],
                "size": obj['Size'],
                "last_modified": obj['LastModified'],
                "provider": "aws"
            })
        
        return files
    
    async def _list_azure_files(self, prefix: str) -> List[Dict]:
        """List files in Azure"""
        if not self.azure_client:
            return []
        
        container_client = self.azure_client.get_container_client(
            self.storage_config["container"]
        )
        
        files = []
        blobs = container_client.list_blobs(name_starts_with=prefix)
        
        for blob in blobs:
            files.append({
                "path": blob.name,
                "size": blob.size,
                "last_modified": blob.last_modified,
                "provider": "azure"
            })
        
        return files
    
    async def _list_local_files(self, prefix: str) -> List[Dict]:
        """List files in local storage"""
        base_path = Path(self.storage_config["path"])
        search_path = base_path / prefix if prefix else base_path
        
        files = []
        if search_path.exists():
            for file_path in search_path.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith('.metadata.json'):
                    relative_path = file_path.relative_to(base_path)
                    files.append({
                        "path": str(relative_path),
                        "size": file_path.stat().st_size,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime),
                        "provider": "local"
                    })
        
        return files
    
    async def _delete_from_s3(self, path: str) -> bool:
        """Delete file from S3"""
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.delete_object(
                Bucket=self.storage_config["bucket"],
                Key=path
            )
            return True
        except Exception:
            return False
    
    async def _delete_from_azure(self, path: str) -> bool:
        """Delete file from Azure"""
        if not self.azure_client:
            return False
        
        try:
            blob_client = self.azure_client.get_blob_client(
                container=self.storage_config["container"],
                blob=path
            )
            blob_client.delete_blob()
            return True
        except Exception:
            return False
    
    async def _delete_locally(self, path: str) -> bool:
        """Delete file from local storage"""
        try:
            local_path = Path(self.storage_config["path"]) / path
            if local_path.exists():
                local_path.unlink()
                
                # Also delete metadata file if exists
                metadata_path = local_path.with_suffix('.metadata.json')
                if metadata_path.exists():
                    metadata_path.unlink()
                
                return True
            return False
        except Exception:
            return False
    
    def _dataframe_to_bytes(self, data: pd.DataFrame, file_format: str) -> bytes:
        """Convert DataFrame to bytes for storage"""
        if file_format == '.parquet':
            return data.to_parquet(index=False)
        elif file_format == '.csv':
            return data.to_csv(index=False).encode('utf-8')
        elif file_format == '.json':
            return data.to_json(orient='records').encode('utf-8')
        else:
            return data.to_parquet(index=False)  # Default to parquet
    
    def _bytes_to_dataframe(self, data_bytes: bytes, file_format: str) -> pd.DataFrame:
        """Convert bytes to DataFrame"""
        if file_format == '.parquet':
            import io
            return pd.read_parquet(io.BytesIO(data_bytes))
        elif file_format == '.csv':
            import io
            return pd.read_csv(io.StringIO(data_bytes.decode('utf-8')))
        elif file_format == '.json':
            import io
            return pd.read_json(io.StringIO(data_bytes.decode('utf-8')))
        else:
            import io
            return pd.read_parquet(io.BytesIO(data_bytes))  # Default to parquet
    
    def _encrypt_sensitive_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encrypt sensitive columns in the data"""
        sensitive_columns = [
            'household_id', 'latitude', 'longitude', 
            'field_officer_id', 'device_id'
        ]
        
        for col in sensitive_columns:
            if col in data.columns:
                data[col] = data[col].astype(str).apply(
                    lambda x: self.cipher_suite.encrypt(x.encode()).decode() if pd.notna(x) else x
                )
        
        return data
    
    def _decrypt_sensitive_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Decrypt sensitive columns in the data"""
        sensitive_columns = [
            'household_id', 'latitude', 'longitude', 
            'field_officer_id', 'device_id'
        ]
        
        for col in sensitive_columns:
            if col in data.columns:
                try:
                    data[col] = data[col].apply(
                        lambda x: self.cipher_suite.decrypt(x.encode()).decode() if pd.notna(x) else x
                    )
                except Exception:
                    # If decryption fails, assume data is not encrypted
                    pass
        
        return data
    
    def _generate_storage_metadata(self, data: pd.DataFrame, path: str) -> Dict:
        """Generate metadata for stored data"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "record_count": len(data),
            "column_count": len(data.columns),
            "columns": list(data.columns),
            "data_types": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "file_path": path,
            "data_hash": self._calculate_data_hash(data),
            "storage_version": "1.0"
        }
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of the data for integrity checking"""
        data_string = data.to_string(index=False)
        return hashlib.md5(data_string.encode()).hexdigest()


class DataArchiver:
    """Handle data archiving for long-term storage"""
    
    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager
    
    async def archive_old_data(self, cutoff_date: datetime) -> Dict:
        """Archive data older than cutoff date"""
        try:
            # List all files
            all_files = await self.storage_manager.list_files("raw/")
            
            archived_count = 0
            archived_size = 0
            
            for file_info in all_files:
                if file_info["last_modified"] < cutoff_date:
                    # Move to archive location
                    archive_path = f"archive/{file_info['path']}"
                    
                    # Retrieve, store in archive, then delete original
                    data = await self.storage_manager.retrieve_data(file_info["path"])
                    await self.storage_manager.store_data(data, archive_path)
                    await self.storage_manager.delete_data(file_info["path"])
                    
                    archived_count += 1
                    archived_size += file_info["size"]
            
            return {
                "archived_files": archived_count,
                "archived_size_bytes": archived_size,
                "archive_date": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error("Data archiving failed", error=str(e))
            raise 