FROM srcd/ml-core

ADD setup.py package
ADD sourced package/sourced
RUN pip3 install --no-cache-dir ./package && rm -rf package

EXPOSE 8000

ENTRYPOINT ["srcml"]
