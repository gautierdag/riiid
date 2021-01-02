import hydra


def get_wd():
    try:
        return hydra.utils.get_original_cwd() + "/"
    except ValueError:
        return ""
