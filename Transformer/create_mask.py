import tensorflow as tf

# Masking
#padding mask
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

#look ahead mask
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def combind_mask(padding_mask, look_ahead_mask):
    return tf.maximum(padding_mask, look_ahead_mask)

if __name__ == "__main__":
    print("This is for mask creating")
    print("Create padding mask: use create_padding_mask(seq)")
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    print("sample input:", x)
    pad = create_padding_mask(x)
    print("result: ", pad)

    print("This is for mask creating")
    print("create look ahead mask: use create_look_ahead_mask(size)")
    x = tf.random.uniform((3, 5))
    print("sample input:", x)
    lam = create_look_ahead_mask(x.shape[1])
    print("result: ", lam)

    print("This is for combinding mask")
    print("for combind masks: combind_mask(padding_mask, look_ahead_mask)")
    print("result: ",combind_mask(pad, lam))