class FakeRequest:
    def __init__(self, content):
        self.content = content

    @property
    def headers(self):
        return {"content-length": len(self.content)}

    def iter_content(self, chunk_size):
        return [self.content[i:i+chunk_size]
                for i in range(0, len(self.content), chunk_size)]


class FakeRequests:
    def __init__(self, router):
        self.router = router

    def get(self, url, params=None, **kwargs):
        return FakeRequest(self.router(url))
