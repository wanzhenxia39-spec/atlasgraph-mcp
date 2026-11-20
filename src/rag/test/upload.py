import requests
import json
import os
import time
import urllib.parse
from typing import Optional

# 配置服务端信息
BASE_URL = "http://0.0.0.0:21032"
UPLOAD_ENDPOINT = f"{BASE_URL}/api/upload"
STATUS_ENDPOINT = f"{BASE_URL}/api/status/"

# 配置用户和数据库信息
USERNAME = "root"  # 实际用户名
DATABASE = "project"  # 目标数据库名
TAR_FILE_NAME = "upload.tar"  # 当前目录下已压缩好的TAR包文件名

# 图模式定义 - 根据您的实际需求定制
DEFAULT_SCHEMA = {
    "vertices": {
        "Entity": ["keyword", "description"],
    },
    "edges": {
        "Relation": {
            "relations": [{"from_key": "Entity", "to_key": "Entity"}],
            "properties": ["keyword"],
        }
    },
}

def upload_and_index() -> Optional[str]:
    # 准备完整的上传配置
    upload_config = {
        "username": USERNAME,
        "database": DATABASE,
        "graph_schema": DEFAULT_SCHEMA,  # 添加图模式定义
        "splitter_mode": "length",     # 使用长度分割模式
        "chunk_size": 512,             # 文本块大小
        "min_chunk_size": 1024,        # 最小文本块大小
        "max_chunk_size": 8192,        # 最大文本块大小
        "chunk_overlap": 100,          # 文本块重叠大小
        "max_workers": 4,              # 最大工作线程数
        "num_extract": 3,              # 实体提取数量
        "parse_pdf": False,            # 是否解析PDF
        "entity_align": True,          # 是否进行实体对齐
        "threashold": 0.85             # 实体对齐阈值
    }
    
    # 将配置转换为JSON字符串并进行URL编码
    config_json = json.dumps(upload_config)
    encoded_config = urllib.parse.quote(config_json)
    
    # 构造正确的URL查询参数
    url_with_params = f"{UPLOAD_ENDPOINT}?upload_config_form={encoded_config}"
    
    # 发送上传请求
    with open(TAR_FILE_NAME, 'rb') as f:
        files = {'upload_file': (TAR_FILE_NAME, f, 'application/x-tar')}
        response = requests.post(url_with_params, files=files)
    
    if response.status_code != 200:
        print(f"上传失败: {response.text}")
        return None
    
    # 获取传输ID
    result = response.json()
    transmission_id = result['transmission_id']
    print(f"文档已上传，传输ID: {transmission_id}")
    
    return transmission_id

def check_processing_status(transmission_id: str) -> bool:
    """轮询任务处理状态"""
    while True:
        response = requests.get(f"{STATUS_ENDPOINT}{transmission_id}")
        if response.status_code != 200:
            print(f"状态检查失败: {response.text}")
            return False
        
        status_info = response.json()
        print(f"当前状态: {status_info['status']}, 进度: {status_info['rate']}%")
        
        if status_info['status'] == "Complete":
            print("知识图谱构建完成！") # 文件上传完成
            return True
        elif status_info['status'] == "failed":
            print("知识图谱构建失败！") # 文件上传失败
            return False
        
        time.sleep(5)  # 每5秒检查一次进度

if __name__ == "__main__":
    trans_id = upload_and_index()
    if trans_id:
        check_processing_status(trans_id)