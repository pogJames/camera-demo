"""uEye camera REST client: event-recording trigger + video fetch. See CLAUDE.md."""

from urllib.parse import urlsplit

import requests
from requests.auth import HTTPBasicAuth

import config


def _base():
    p = urlsplit(config.STREAM_URL)
    return f"{p.scheme}://{p.netloc}"


def _auth():
    return HTTPBasicAuth(config.CAM_USER, config.CAM_PASS) if config.CAM_USER else None


def enable_recording(log=print):
    try:
        r = requests.patch(
            f"{_base()}/recording/eventbased",
            data={"Enabled": "true", "RecordedStream": config.RECORD_STREAM,
                  "PreEventTime": config.RECORD_PRE_SECS,
                  "PostEventTime": config.RECORD_POST_SECS},
            auth=_auth(), verify=config.VERIFY_TLS, timeout=5)
        r.raise_for_status()
        log(f"[camera] event recording enabled on {config.RECORD_STREAM}")
    except Exception as e:
        log(f"[camera] enable recording failed: {e!r}")


def trigger(origin):
    r = requests.post(
        f"{_base()}/recording/eventbased", params={"EventOrigin": origin},
        auth=_auth(), verify=config.VERIFY_TLS, timeout=3)
    r.raise_for_status()
    return r.text.strip().strip('"') or None


def open_video(name):
    return requests.get(
        f"{_base()}/recording/videos/list/{name}",
        auth=_auth(), verify=config.VERIFY_TLS, stream=True, timeout=(5, 30))
