FROM ubuntu:16.04

ADD requirements.txt setup.py package/
ADD ast2vec package/ast2vec
ADD doc/how_to_use_ast2vec.ipynb /home

RUN rm -rf package/ast2vec/tests && \
    apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends \
        ca-certificates locales git python3 python3-dev libxml2 libxml2-dev \
        libonig2 make gcc curl && \
    curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip3 install --no-cache-dir -r package/requirements.txt jupyter && \
    apt-get remove -y python3-dev libxml2-dev make gcc curl && \
    apt-get remove -y *-doc *-man >/dev/null && \
    apt-get autoremove -y && \
    apt-get install -y --no-install-suggests --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8 && \
    mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.token = ''" > ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir = '/home'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo '#!/bin/bash\n\
\n\
echo\n\
echo "	$@"\n\
echo\n\' > /browser && \
    chmod +x /browser


RUN pip3 install --no-cache-dir ./package && rm -rf package

EXPOSE 8000
ENV BROWSER /browser
ENV LC_ALL en_US.UTF-8

ENTRYPOINT ["ast2vec"]
