'''
Created on Feb 10, 2026

@author: TMARTI02
'''


import requests
from requests.exceptions import ConnectionError, Timeout, RequestException


from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

    
def append_query(base_url: str, params: dict, force_trailing_slash: bool = False) -> str:
    """
    Append query params to base_url, optionally forcing a trailing '/' in the path.
    Ensures format like: https://.../search/?typeId=3&modelId=1065
    """
    parsed = urlparse(str(base_url))
    path = parsed.path or ""
    if force_trailing_slash and path and not path.endswith("/"):
        path = path + "/"

    # Merge existing + new query params
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    for k, v in params.items():
        pairs.append((k, str(v)))
    new_query = urlencode(pairs)

    new_parsed = parsed._replace(path=path, query=new_query)
    return urlunparse(new_parsed)
  
    

def is_working_image_link(url: str) -> bool:
    """
    Checks if a given URL is a working link and points to a non-empty image.

    Returns True only if:
      - The URL is accessible (2xx),
      - Content-Type starts with 'image/',
      - And the body is not empty (Content-Length > 0 or at least one data chunk).
    """
    timeout = 5
    try:
        # Try HEAD first to avoid downloading the body
        head = requests.head(url, allow_redirects=True, timeout=timeout)

        if 200 <= head.status_code < 300:
            content_type = (head.headers.get("Content-Type") or "").lower()
            if not content_type.startswith("image/"):
                print(f"URL is accessible but content type is not an image: {content_type}")
                return False

            # If server reports content length, reject zero-length
            cl = head.headers.get("Content-Length")
            if cl is not None:
                try:
                    if int(cl) == 0:
                        print("Blank image (Content-Length is 0).")
                        return False
                    else:
                        # Non-zero length reported: consider it valid without GET
                        return True
                except ValueError:
                    # Content-Length not an integer; fall through to GET check
                    pass

            # No reliable Content-Length; verify there's at least some body via GET
            with requests.get(url, allow_redirects=True, timeout=timeout, stream=True) as resp:
                if not (200 <= resp.status_code < 300):
                    print(f"URL not working. Status code (GET): {resp.status_code}")
                    return False

                ctype2 = (resp.headers.get("Content-Type") or "").lower()
                if not ctype2.startswith("image/"):
                    print(f"GET returned non-image content type: {ctype2}")
                    return False

                # Read a small chunk to ensure the body isn't empty
                try:
                    first_chunk = next(resp.iter_content(chunk_size=1024), b"")
                except StopIteration:
                    first_chunk = b""

                if not first_chunk:
                    print("Blank image (no bytes in response body).")
                    return False

                return True

        else:
            # HEAD failed; try GET with stream=True and validate minimal body
            with requests.get(url, allow_redirects=True, timeout=timeout, stream=True) as resp:
                if not (200 <= resp.status_code < 300):
                    print(f"URL not working. Status code (GET): {resp.status_code}")
                    return False

                content_type = (resp.headers.get("Content-Type") or "").lower()
                if not content_type.startswith("image/"):
                    print(f"GET returned non-image content type: {content_type}")
                    return False

                try:
                    first_chunk = next(resp.iter_content(chunk_size=1024), b"")
                except StopIteration:
                    first_chunk = b""

                if not first_chunk:
                    print("Blank image (no bytes in response body).")
                    return False

                return True

    except ConnectionError:
        print(f"Connection error for URL: {url}")
    except Timeout:
        print(f"Timeout error for URL: {url}")
    except RequestException as e:
        print(f"An error occurred during the request: {e}")

    return False

def test_working_link():
    
    # --- Example Usage ---
    # A known working image link
    # working_url = "https://images.pexels.com" 
    # # A known broken link
    # broken_url = "https://not.a.real.domain"
    # # A working link that is not an image (e.g., Google's homepage)
    # not_image_url = "https://www.google.com"
    # print(f"Checking {working_url}: {is_working_image_link(working_url)}")
    # print(f"Checking {broken_url}: {is_working_image_link(broken_url)}")
    # print(f"Checking {not_image_url}: {is_working_image_link(not_image_url)}")
    
    image_url_ctx_ok="https://ctx-api-dev.ccte.epa.gov/chemical/property/model/file/search/?typeId=3&modelId=1065"
    image_url_ctx_missing="https://ctx-api-dev.ccte.epa.gov/chemical/property/model/file/search/?typeId=3&modelId=9999"
    
    print(f"Checking ok_ctx_api_image: {is_working_image_link(image_url_ctx_ok)}")
    print(f"Checking missing_ctx_api_image: {is_working_image_link(image_url_ctx_missing)}")

if __name__ == '__main__':
    test_working_link()