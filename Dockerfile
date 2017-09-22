FROM ubuntu:16.04

ADD requirements.txt setup.py package/
ADD ast2vec package/ast2vec

RUN rm -rf package/ast2vec/tests && \
    apt-get update && \
    apt-get install -y locales git python3 python3-dev libxml2 libxml2-dev make gcc curl && \
    apt-get clean && \
    curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip3 install --no-cache-dir -r package/requirements.txt && \
    apt-get remove -y python3-dev libxml2-dev make gcc curl && \
    apt-get remove -y *-doc *-man && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8 && \
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
