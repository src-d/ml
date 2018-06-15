FROM srcd/spark:2.2.1

ADD requirements.txt package/

RUN rm -rf package/sourced/ml/tests && \
    apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends ca-certificates locales \
      git libxml2 libxml2-dev libsnappy1 libsnappy-dev make gcc g++ && \
    pip3 install --no-cache-dir -r package/requirements.txt && \
    pip3 uninstall sourced-engine -y && \
    apt-get remove -y python3-dev libxml2-dev libsnappy-dev make gcc g++ curl && \
    apt-get remove -y .*-doc .*-man >/dev/null && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    echo '#!/bin/bash\n\
\n\
echo\n\
echo "	$@"\n\
echo\n\' > /browser && \
    chmod +x /browser

ENV BROWSER=/browser \
    LC_ALL=en_US.UTF-8 
