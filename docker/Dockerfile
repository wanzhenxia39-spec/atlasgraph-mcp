# Supply a compatible base image with --build-arg BASE_IMAGE=...
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

#root 文件夹包含atlasgraph安装包，requirements.txt，及最新RAG代码
#usr 内为字体文件
COPY ./docker/tmpfiles_root /root
COPY ./docker/tmpfiles_usr /usr/share/fonts/songti

#安装依赖
RUN pip install atlasgraph-1.1.0.1-py3-none-any.whl;\
    rm atlasgraph-1.1.0.1-py3-none-any.whl; \
    pip install -i https://pypi.doubanio.com/simple -r requirements_incr.txt
COPY . /root/rag
ENV  PYTHONPATH=/root/rag/src:$PYTHONPATH
