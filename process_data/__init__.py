__all__ = [
    "FaultDetectionDataset",
    "InterpretationDataset",
    "build_fault_detection_dataset",
    "build_interpretation_dataset",
    "load_saved_dataset",
]


def __getattr__(name):
    if name in __all__:
        from process_data import torch_dataset

        return getattr(torch_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
