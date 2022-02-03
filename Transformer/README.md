# Digest Transformer model from official TensorFlow website

## Reference
https://www.tensorflow.org/text/tutorials/transformer

## File list
### Jupyter notebook:
Transformer.ipynb

Which include my study notes and functional test.

### Python files about constructing and training Transformer model
1. Position Encoding
	
	position_encoding.py
	
1. Multi Head Attention (MHA)
   
   multi\_head\_attention.py
   <figure>
   <img src=images/transformer/mha_img_original.png width=400 />
   </figure>
1. Create look ahead and padding masks
	
	create_mask.py
	
1. Use (1, 2, 3) to assemble the Encoding\Decoding unit layer

	unit_layers.py

1. Encoder/Decoder

	modules.py

1. Transformer model

	transformer.py

### Training steps
1. Learning rate

```math
lrate = d^{-0.5}_{model}*min(step_num^{-0.5}, step\_ num \cdot warmup\_ steps^{-1.5})
```

	
	