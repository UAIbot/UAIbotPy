import sys


def url_modified(url, port):
    if port == "":
        return url
    else:
        base = "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master"
        relative_path = url[len(base) :].lstrip("/")
        return f"http://localhost:{port}/{relative_path}"


def is_url_available(url, types):
    """
    Try to access the content of the url 'url'. Also verifies if the content is one of the extensions contained in
    'types' (e.g., types = ['png', 'bmp', 'jpg', 'jpeg'] for images).

    Never throws an Exception, always returning a string with a message. Returns 'ok!' if and only if the
    url was succesfully acessed and has the correct file type.

    Parameters
    ----------
    url: string
        The url string.

    types: list of string
        The desired content extensions.

    Returns
    -------
    message: string
        Message.
    """

    if not isinstance(url, str):
        return " is not a valid url"

    ind = url.rfind(".")
    filetype = url[ind + 1 :]

    if not (filetype in types):
        return " must be an file of the types: '" + str(types) + "'"

    try:
        import httplib2

        h = httplib2.Http(disable_ssl_certificate_validation=True)  # Change this later
        resp = h.request(url, "HEAD")
        if int(resp[0]["status"]) < 400:
            return "ok!"
        else:
            return " : not able to access '" + url + "'."
    except:
        return " : not able to access '" + url + "'."


def get_environment():
    try:
        import google.colab

        return "Colab"
    except ImportError:
        try:
            from IPython import get_ipython

            if "google.colab" in sys.modules:
                return "Colab"
            elif get_ipython() is not None:
                return "Jupyter"
        except ImportError:
            return "Not in Jupyter/Colab"
    return "Local"
