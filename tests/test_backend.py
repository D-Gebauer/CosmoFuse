from CosmoFuse.backend import get_backend
import numpy as np

def test_backend_cpu():
    backend = get_backend('cpu')
    assert backend.name == 'numpy'
    arr = backend.asarray([1, 2, 3])
    assert isinstance(arr, np.ndarray)

def test_backend_auto():
    # Simulate cupy presence or absence
    try:
        import cupy
        expected = 'cupy'
    except ImportError:
        expected = 'numpy'

    backend = get_backend('auto')
    assert backend.name == expected

def test_backend_explicit_gpu_no_cupy(monkeypatch):
    import sys
    # Mock cupy absence
    monkeypatch.setitem(sys.modules, 'cupy', None)

    try:
        get_backend('gpu')
    except ImportError:
        pass  # Expected
    except Exception as e:
         # Depending on implementation details, it might raise ValueError or ImportError
         assert "Cupy not installed" in str(e)

if __name__ == "__main__":
    test_backend_cpu()
    test_backend_auto()
    print("Backend tests passed!")
