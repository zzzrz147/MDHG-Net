import tensorflow as tf


def build_role_prior(feature_names, role_groups, num_edges):
    groups = list(role_groups.values())
    if not groups:
        raise ValueError('At least one role group is required.')
    return tf.constant([[float(name in groups[m % len(groups)]) for m in range(num_edges)]
                        for name in feature_names], dtype=tf.float32)
