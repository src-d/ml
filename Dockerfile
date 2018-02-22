FROM srcd/spark:2.2.0_v2

ADD requirements.txt setup.py package/

RUN rm -rf package/sourced/ml/tests && \
    apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends ca-certificates locales \
      git python3 python3-dev libxml2 libxml2-dev libonig2 libsnappy-dev make gcc g++ curl && \
    curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip3 install --no-cache-dir -r package/requirements.txt && \
    apt-get remove -y python3-dev libxml2-dev make gcc g++ curl && \
    apt-get remove -y .*-doc .*-man >/dev/null && \
    apt-get autoremove -y && \
    apt-get install -y --no-install-suggests --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8 && \
    echo '#!/bin/bash\n\
\n\
echo\n\
echo "	$@"\n\
echo\n\' > /browser && \
    chmod +x /browser


ADD sourced package/sourced
RUN pip3 install --no-cache-dir ./package && rm -rf package

EXPOSE 8000
ENV BROWSER /browser
ENV LC_ALL en_US.UTF-8
ENV PYSPARK_PYTHON python3
ENV PYSPARK_PYTHON_DRIVER python3

ENTRYPOINT ["srcml"]
