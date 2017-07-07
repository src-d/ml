import argparse
import json
import os
import unittest

from google.cloud.storage import Client

from ast2vec import Id2Vec, NBOW, DocumentFrequencies, setup_logging
from ast2vec.publish import list_models, publish_model
from ast2vec.tests.test_dump import captured_output, paths


class PublishTests(unittest.TestCase):
    BUCKET = "test-ast2vec"
    CREDENTIALS = os.path.join(os.path.dirname(__file__), "..", "..", "gcs.json")
    ACCESS = os.path.exists(CREDENTIALS)
    REASON = "No access to Google Cloud Storage"

    @classmethod
    def setUpClass(cls):
        setup_logging("INFO")

    @unittest.skip("GCS service account permissions are not ready")
    @unittest.skipIf(not ACCESS, REASON)
    def test_list(self):
        args = argparse.Namespace(gcs=self.BUCKET)
        with captured_output() as (out, _, _):
            list_models(args)
        out = out.getvalue().split("\n")
        for name in (Id2Vec.NAME, NBOW.NAME, DocumentFrequencies.NAME):
            idx = out.index(name)
            self.assertGreaterEqual(idx, 0)
            while idx < len(out):
                idx += 1
                if out[idx].startswith("  * "):
                    break
                else:
                    self.assertEqual(out[idx][:4], "    ")
            else:
                self.fail("The default model was not found.")

    @unittest.skip("GCS service account permissions are not ready")
    @unittest.skipIf(not ACCESS, REASON)
    def test_publish(self):
        client = Client.from_service_account_json(self.CREDENTIALS)
        bucket = client.get_bucket(self.BUCKET)
        blob = bucket.get_blob(Id2Vec.INDEX_FILE)
        backup = blob.download_as_string()
        index = json.loads(backup.decode("utf-8"))
        del index["models"]["id2vec"]["92609e70-f79c-46b5-8419-55726e873cfc"]
        del index["models"]["id2vec"]["default"]
        updated = json.dumps(index, indent=4, sort_keys=True)
        blob.upload_from_string(updated)
        try:
            args = argparse.Namespace(
                model=os.path.join(os.path.dirname(__file__), paths.ID2VEC),
                gcs=self.BUCKET, update_default=True, force=False,
                credentials=self.CREDENTIALS)
            with captured_output() as (out, err, log):
                result = publish_model(args)
            self.assertEqual(result, 1)
            self.assertIn("Model 92609e70-f79c-46b5-8419-55726e873cfc already "
                          "exists, aborted", log.getvalue())
            blob = bucket.get_blob(
                "models/id2vec/92609e70-f79c-46b5-8419-55726e873cfc.asdf")
            bucket.rename_blob(
                blob,
                "models/id2vec/92609e70-f79c-46b5-8419-55726e873cfc.asdf.bak")
            try:
                with captured_output() as (out, err, log):
                    result = publish_model(args)
                blob = bucket.get_blob(
                    "models/id2vec/92609e70-f79c-46b5-8419-55726e873cfc.asdf")
                self.assertTrue(blob.exists())
                blob.delete()
                self.assertIsNone(result)
                self.assertIn("Uploaded as ", log.getvalue())
                self.assertIn("92609e70-f79c-46b5-8419-55726e873cfc", log.getvalue())
            finally:
                blob = bucket.get_blob(
                    "models/id2vec/92609e70-f79c-46b5-8419-55726e873cfc.asdf.bak")
                bucket.rename_blob(
                    blob, "models/id2vec/92609e70-f79c-46b5-8419-55726e873cfc.asdf")
                blob = bucket.get_blob(
                    "models/id2vec/92609e70-f79c-46b5-8419-55726e873cfc.asdf")
                blob.make_public()
            blob = bucket.get_blob(Id2Vec.INDEX_FILE)
            self.assertTrue(blob.download_as_string(), backup)
        finally:
            blob = bucket.get_blob(Id2Vec.INDEX_FILE)
            blob.upload_from_string(backup)
            blob.make_public()


if __name__ == "__main__":
    unittest.main()
