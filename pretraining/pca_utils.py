import torch


def pca_covariance(W, n_components):
    # PCA on covariance matrix
    U, S, V = torch.pca_lowrank(W @ W.T, q=n_components, center=False)

    # expansion/reduction matrix
    E = torch.sqrt(torch.diag(S)) @ U.T
    E_inv = U @ torch.linalg.pinv(torch.sqrt(torch.diag(S)))  # TODO: check if V -> U makes sense

    return E, E_inv


def pca_weight(W, n_components):
    # PCA on weight matrix
    U, S, V = torch.pca_lowrank(W, q=n_components, center=False)

    # expansion/reduction matrix
    E = U.T
    E_inv = E.T

    # # output weight: alternative
    # W_out_expand = torch.diag(S) @ V.T

    return E, E_inv


def expand_out_pca_diag(W_diag, d_out_new):
    # dimension check
    d_out_double, d_in_double = W_diag.shape
    assert d_out_double >= d_out_new > (d_out_double / 2)

    if min(d_out_double, d_in_double) <= d_out_new:
        print(f"PCA on covariance matrix, since min({d_out_double}, {d_in_double}) <= {d_out_new}")
        E, E_inv = pca_covariance(W_diag, n_components=d_out_new)
    else:
        # NOTE: this can be done with covariance matrix as well
        print("PCA on weight matrix")
        E, E_inv = pca_weight(W_diag, n_components=d_out_new)

    # output weight
    W_out_expand = E @ W_diag

    print(f"Mean reconstruction error: {torch.abs(E_inv @ W_out_expand - W_diag).mean()}")

    # output dimension sanity check
    assert W_out_expand.shape == (d_out_new, d_in_double)
    assert E_inv.shape == (d_out_double, d_out_new)

    return E, E_inv


def expand_out_pca_separately(A, B, d_out_new):
    # dimension check
    d_out, d_in = A.shape
    assert A.shape == B.shape
    assert d_out * 2 >= d_out_new > d_out

    d_out_new_half = d_out_new // 2

    Es = []
    E_invs = []
    for W in (A, B):
        if min(d_out, d_in) <= d_out_new_half:
            print(f"PCA on covariance matrix, since min({d_out}, {d_in}) <= {d_out_new_half}")
            E, E_inv = pca_covariance(W, n_components=d_out_new_half)
        else:
            print("PCA on weight matrix")
            # NOTE: this can be done with covariance matrix as well
            E, E_inv = pca_weight(W, n_components=d_out_new_half)

            # output weight
            W_out_expand = E @ W

            print(f"Mean reconstruction error: {torch.abs(E_inv @ W_out_expand - W).mean()}")

        # print(W_out_expand.shape, E_inv.shape)

        Es.append(E)
        E_invs.append(E_inv)

    return Es, E_invs
