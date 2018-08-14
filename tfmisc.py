import tensorflow as tf


def paramterCount(var_list=None):
    import numpy as np
    if var_list is None:
        var_list = tf.trainable_variables()

    return np.sum([np.prod(v.shape) for v in var_list])


def selectFromRows(tensor, indexes):
    row = tf.range(tf.shape(indexes)[0])
    return tf.gather_nd(tensor, tf.stack([row, indexes], axis=1))
    # shp = tf.shape(tensor)
    # indexes_flat = tf.range(shp[0]) * shp[1] + indexes
    # return tf.gather(tf.reshape(tensor, [-1]), indexes_flat)


def policyGradientLoss(action, policy_gradient):
    loss = tf.log(action) * policy_gradient
    loss = -tf.reduce_sum(loss)
    return loss


def entropyLoss(tensor):
    entropy = -tf.reduce_sum(tensor * tf.log(tensor), 1, name="entropy")
    return tf.reduce_mean(entropy, name="entropy_mean")

# NONE GRADS/VARS MIGHT BE NEEDED:
# def getGradAndVars(optimizer, loss):
#     grads_and_vars = optimizer.compute_gradients(loss)
#     grads_and_vars = [[grad, var]
#                       for grad, var in grads_and_vars if grad is not None]
#     return grads_and_vars


def clipGrads(optimizer, loss, clip):
    grads_and_vars = optimizer.compute_gradients(
        loss)  # getGradAndVars(optimizer, loss)
    clipped = []
    for g, v in grads_and_vars:
        if g is not None:
            clipped.append((tf.clip_by_value(g, -clip, clip), v))
        else:
            clipped.append((g, v))
    return clipped


def clipedOptimize(optimizer, loss, clip, global_step=None):
    gv = clipGrads(optimizer, loss, clip)
    return optimizer.apply_gradients(gv, global_step=global_step)


def applyGradients(optimizer, from_grads, to_vars, clip_ratio=None):
    if clip_ratio:  # TODO: fix the error when True
        from_grads = tf.clip_by_global_norm(from_grads, clip_ratio)
    combined_grads_and_vars = zip(from_grads, to_vars)
    return optimizer.apply_gradients(combined_grads_and_vars)


# def transferLearning(optimizer, from_loss, to_loss, clip_ratio=None):
#     from_grads_and_vars = getGradAndVars(optimizer, from_loss)
#     grads = [grad for grad, var in from_grads_and_vars if grad is not None]

#     to_grads_and_vars = getGradAndVars(optimizer, to_loss)
#     variables = [var for grad, var in to_grads_and_vars]

#     return applyGradients(optimizer, grads, variables, clip_ratio)


def getScopeParameters(scope_name):
    if tf.get_variable_scope().name:
        scope = tf.get_variable_scope().name + '/'
    else:
        scope = ''
    variables = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + scope_name)
    print("There are %d variables in `%s`" %
          (len(variables), scope + scope_name))
    assert len(variables) > 0
    return variables


def copyScopeVars(from_scope, to_scope, tau=None):
    from_list = getScopeParameters(from_scope)
    target_list = getScopeParameters(to_scope)

    if tf.get_variable_scope().name:
        scope = tf.get_variable_scope().name + '/'
    else:
        scope = ''

    from_scope = scope + from_scope
    to_scope = scope + to_scope

    from_list = sorted(from_list, key=lambda v: v.name)
    target_list = sorted(target_list, key=lambda v: v.name)

    assert len(from_list) == len(target_list)

    print("Copying %d variables from %s to %s" %
          (len(from_list), from_scope, to_scope))

    operations = []
    for i in range(len(from_list)):
        assert target_list[i].name[len(
            to_scope):] == from_list[i].name[len(from_scope):], "Incopatible names %s and %s" % (target_list[i].name[len(
                to_scope):], from_list[i].name[len(from_scope):])
        if tau is not None:
            new_value = tf.multiply(
                from_list[i], tau) + tf.multiply(target_list[i], (1 - tau))
        else:
            print(from_list[i].name, from_list[i].get_shape().as_list(), '->', target_list[i].name,
                  target_list[i].get_shape().as_list())
            new_value = from_list[i]

        operations.append(target_list[i].assign(new_value))

    return operations
