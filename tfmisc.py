import tensorflow as tf


def selectFromRows(tensor, indexes):
    shp = tf.shape(tensor)
    indexes_flat = tf.range(0, shp[0]) * shp[1] + indexes
    return tf.gather(tf.reshape(tensor, [-1]), indexes_flat)


def policyGradientLoss(action, policy_gradient):
    loss = tf.log(action) * policy_gradient
    loss = -tf.reduce_sum(loss)
    return loss


def entropyLoss(tensor):
    entropy = -tf.reduce_sum(tensor * tf.log(tensor), 1, name="entropy")
    return tf.reduce_mean(entropy, name="entropy_mean")


def getGradAndVars(optimizer, loss):
    grads_and_vars = optimizer.compute_gradients(loss)
    grads_and_vars = [[grad, var]
                      for grad, var in grads_and_vars if grad is not None]
    return grads_and_vars


def applyGradients(optimizer, from_grads, to_vars, clip_ratio=None):
    if clip_ratio:  # TODO: fix the error when True
        from_grads = tf.clip_by_global_norm(from_grads, clip_ratio)
    combined_grads_and_vars = zip(from_grads, to_vars)
    return optimizer.apply_gradients(combined_grads_and_vars)


def transferLearning(optimizer, from_loss, to_loss, clip_ratio=None):
    from_grads_and_vars = getGradAndVars(optimizer, from_loss)
    grads = [grad for grad, var in from_grads_and_vars]

    to_grads_and_vars = getGradAndVars(
        optimizer, to_loss)
    variables = [var for grad, var in to_grads_and_vars]

    return applyGradients(
        optimizer, grads, variables, clip_ratio)


def getScopeParameters(scope_name):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)


def copyScopeVars(from_scope, to_scope, tau=None):
    if tf.get_variable_scope().name:
        scope = tf.get_variable_scope().name + '/'
    else:
        scope = ''

    from_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + from_scope)
    target_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + to_scope)

    from_list = sorted(from_list, key=lambda v: v.name)
    target_list = sorted(target_list, key=lambda v: v.name)

    assert len(from_list) == len(target_list)
    assert len(target_list) > 0

    operations = []
    for i in range(len(from_list)):
        if tau is not None:
            new_value = tf.multiply(
                from_list[i], tau) + tf.multiply(target_list[i], (1 - tau))
        else:
            new_value = from_list[i]

        operations.append(target_list[i].assign(new_value))

    return operations

