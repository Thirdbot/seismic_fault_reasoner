# plan for model architecture

## using 2 dataset
    1. seismic_fault_detection (GN1101_2012 and its fault labels)
    2. seismic_interpretation (extracted multimodal data from reports PDF)

## The Idea
Simply to train vision encoder to be fault-aware first (using seismic_fault_detection dataset) , then align encoder's output in share's layers (multimodal layers) , finally get text and images output

### architecture would be:
    raw seismic image
      -> vision encoder
          should encode: reflectors, discontinuities, fault offsets, fault planes
      -> fusion/alignment layers
          should map visual fault evidence to text concepts
      -> text output / image output
          explains fault interpretation using report language with its image output.
### finetuning options
publicly available geoscience models (text-based)
    https://huggingface.co/geobrain-ai/geogalactica
### train stages
- first, train the vision encoder to be fault-awares by using seismic_fault_detection dataset . Output would be fault mask / fault heatmap / fault presence
- second, pass-in image and text to train the text decoder (freeze encoder) to be context-awares by using seismic_interpretation dataset. Output would be explanation

### evaluation stages
use validation set from seismic_interpretation dataset as evaluate dataset
