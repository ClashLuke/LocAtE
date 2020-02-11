import os


def install_tpu():
    if os.environ.get('COLAB_TPU_ADDR', False):
        os.environ['TRIM_GRAPH_SIZE'] = "500000"
        os.environ['TRIM_GRAPH_CHECK_FREQUENCY'] = "20000"

        if not os.path.exists('/content/torchvision-1.15-cp36-cp36m-linux_x86_64.whl'):
            import collections
            from datetime import datetime, timedelta
            import requests
            import threading

            _VersionConfig = collections.namedtuple('_VersionConfig', 'wheels,server')
            VERSION = "xrt==1.15.0"  # @param ["xrt==1.15.0", "torch_xla==nightly"]
            CONFIG = {
                    'xrt==1.15.0': _VersionConfig('1.15', '1.15.0'),
                    'torch_xla==nightly': _VersionConfig('nightly', 'XRT-dev{}'.format(
                            (datetime.today() - timedelta(1)).strftime('%Y%m%d'))),
                    }[VERSION]
            DIST_BUCKET = 'gs://tpu-pytorch/wheels'
            TORCH_WHEEL = 'torch-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
            TORCH_XLA_WHEEL = 'torch_xla-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
            TORCHVISION_WHEEL = 'torchvision-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)

            # Update TPU XRT version
            def update_server_xrt():
                print('Updating server-side XRT to {} ...'.format(CONFIG.server))
                url = 'http://{TPU_ADDRESS}:8475/requestversion/{XRT_VERSION}'.format(
                        TPU_ADDRESS=os.environ['COLAB_TPU_ADDR'].split(':')[0],
                        XRT_VERSION=CONFIG.server,
                        )
                print('Done updating server-side XRT: {}'.format(requests.post(url)))

            update = threading.Thread(target=update_server_xrt)
            update.start()

            # Install Colab TPU compat PyTorch/TPU wheels and dependencies
            os.system("""pip uninstall -y torch torchvision ; gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" . ; """
                      + """gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" . ; gsutil cp "$DIST_BUCKET/$TORCHVISION_WHEEL" . ;"""
                      + """pip install "$TORCH_WHEEL";pip install "$TORCH_XLA_WHEEL";pip install "$TORCHVISION_WHEEL";"""
                      + """sudo apt-get install libomp5""")
            update.join()
