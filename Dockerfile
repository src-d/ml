FROM srcd/ml-core

COPY setup.py package
COPY README.md package
COPY sourced package/sourced
RUN pip3 install --no-cache-dir ./package && rm -rf package

EXPOSE 8000

ENTRYPOINT ["srcml"]
