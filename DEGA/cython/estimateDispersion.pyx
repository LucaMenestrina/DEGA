# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

# cython: np_pythran=True
# cython: cxx=True
# cython: -DUSE_XSIMD -fopenmp -march=native

from scipy.special import gammaln, digamma, polygamma
# works only on scalars (no array broadcasting)
from scipy.special.cython_special cimport gammaln as cgammaln
from scipy.stats import nbinom
import numpy as np
from libc.math cimport log, exp


import cython
cimport numpy as np
np.import_array()


cpdef trigamma(x):
    return polygamma(1, x)

cpdef double trace(double[:, ::1] a):
    cdef int d = min(a.shape[0], a.shape[1])
    cdef Py_ssize_t i
    cdef double t = 0

    for i in xrange(d):
        t += a[i, i]

    return t

cdef double log_posterior(
    double log_alpha,
    np.ndarray[np.int64_t, ndim=1, mode="c"] counts_row,
    np.ndarray[np.float64_t, ndim=1, mode="c"] mu_row,
    designMatrix,
    double log_alpha_prior_mean,
    double log_alpha_prior_sigmasq,
    bint use_prior,
    np.ndarray[np.float64_t, ndim=1, mode="c"] weights_row,
    bint use_weights,
    float weight_threshold,
    bint use_CR,
):
    cdef:
        double alpha = exp(log_alpha)
        double alpha_neg1 = alpha ** -1
        double cr_term
        double ll_part
        double prior_part
        double lp

        double[::1] w_diag

    if use_CR:
        w_diag = pow((pow(mu_row, -1) + alpha), -1)
        if use_weights:
            designMatrix = designMatrix[[weights_row > weight_threshold]]
            designMatrix = designMatrix[
                designMatrix.columns[[abs(designMatrix).sum(axis=0) > 0]]
            ]
            w_diag = w_diag.loc[[weights_row > weight_threshold]]
        cr_term = -0.5 * log(
            np.linalg.det(np.matmul(designMatrix.transpose(),
                          (designMatrix * w_diag[:, None])))
        )
    else:
        cr_term = 0
    if use_weights:
        ll_part = (
            weights_row * (gammaln(counts_row + alpha_neg1))
            - cgammaln(alpha_neg1)
            - counts_row * np.log(mu_row + alpha_neg1)
            - alpha_neg1 * np.log(1 + (mu_row * alpha))
        ).sum()
    else:
        ll_part = (
            gammaln(counts_row + alpha_neg1)
            - cgammaln(alpha_neg1)
            - counts_row * np.log(mu_row + alpha_neg1)
            - alpha_neg1 * np.log(1 + (mu_row * alpha))
        ).sum()
    if use_prior:
        prior_part = (
            -0.5 * ((log_alpha - log_alpha_prior_mean) ** 2)
            / log_alpha_prior_sigmasq
        )
    else:
        prior_part = 0

    lp = ll_part + prior_part + cr_term
    return lp


cdef double dlog_posterior(
    double log_alpha,
    np.ndarray[np.int64_t, ndim=1, mode="c"] counts_row,
    np.ndarray[np.float64_t, ndim=1, mode="c"] mu_row,
    designMatrix,
    double log_alpha_prior_mean,
    double log_alpha_prior_sigmasq,
    bint use_prior,
    np.ndarray[np.float64_t, ndim=1, mode="c"] weights_row,
    bint use_weights,
    float weight_threshold,
    bint use_CR,
):
    cdef:
        double alpha = exp(log_alpha)
        double alpha_neg1 = alpha ** -1
        double alpha_neg2 = alpha ** -2
        double cr_term
        double ll_part
        double prior_part
        double lp

        double[::1] w_diag
        double[::1] dw_diag
        # double[::1] b
        # double detb
        # double[::1] db
        # double detdb

    if use_CR:
        w_diag = pow((pow(mu_row, -1) + alpha), -1)
        dw_diag = -pow((pow(mu_row, -1) + alpha), -2)
        if use_weights:
            designMatrix = designMatrix[[weights_row > weight_threshold]]
            designMatrix = designMatrix[
                designMatrix.columns[[abs(designMatrix).sum(axis=0) > 0]]
            ]
            w_diag = w_diag.loc[[weights_row > weight_threshold]]
            dw_diag = dw_diag.loc[[weights_row > weight_threshold]]
        b = np.matmul(designMatrix.transpose(),
                      (designMatrix * w_diag[:, None]))
        detb = np.linalg.det(b)
        db = np.matmul(designMatrix.transpose(),
                       (designMatrix * dw_diag[:, None]))
        detdb = detb * trace(np.linalg.solve(b, db))
        cr_term = -0.5 * (detdb / detb)
    else:
        cr_term = 0
    if use_weights:
        ll_part = alpha_neg2 * (
            weights_row
            * (
                digamma(alpha_neg1)
                + np.log(1 + mu_row * alpha)
                - mu_row * alpha * pow((1 + mu_row * alpha), -1)
                - digamma(counts_row + alpha_neg1)
                + counts_row * pow((mu_row + alpha_neg1), -1)
            )
        ).sum()
    else:
        ll_part = alpha_neg2 * (
            (
                digamma(alpha_neg1)
                + np.log(1 + mu_row * alpha)
                - mu_row * alpha * pow((1 + mu_row * alpha), -1)
                - digamma(counts_row + alpha_neg1)
                + counts_row * pow((mu_row + alpha_neg1), -1)
            )
        ).sum()
    if use_prior:
        prior_part = -((log_alpha - log_alpha_prior_mean)
                       / log_alpha_prior_sigmasq)
    else:
        prior_part = 0

    lp = (ll_part + cr_term) * alpha + prior_part
    return lp


cdef double d2log_posterior(
    double log_alpha,
    np.ndarray[np.int64_t, ndim=1, mode="c"] counts_row,
    np.ndarray[np.float64_t, ndim=1, mode="c"] mu_row,
    designMatrix,
    double log_alpha_prior_mean,
    double log_alpha_prior_sigmasq,
    bint use_prior,
    np.ndarray[np.float64_t, ndim=1, mode="c"] weights_row,
    bint use_weights,
    float weight_threshold,
    bint use_CR,
):
    cdef:
        double alpha = exp(log_alpha)
        double alpha_neg1 = alpha ** -1
        double alpha_neg2 = alpha ** -2
        double detb
        double ddetb
        double d2detb
        double cr_term
        double ll_part
        double prior_part
        double lp

        double[::1] w_diag
        double[::1] dw_diag
        double[::1] d2w_diag
        # double[::1] b
        # double[::1] db
        # double[:, ::1] d_db

    designMatrix_original = designMatrix
    if use_CR:
        w_diag = pow((pow(mu_row, -1) + alpha), -1)
        dw_diag = -pow((pow(mu_row, -1) + alpha), -2)
        d2w_diag = -2 * pow((pow(mu_row, -1) + alpha), -3)
        if use_weights:
            designMatrix = designMatrix[[weights_row > weight_threshold]]
            designMatrix = designMatrix[
                designMatrix.columns[[abs(designMatrix).sum(axis=0) > 0]]
            ]
            w_diag = w_diag.loc[[weights_row > weight_threshold]]
            dw_diag = dw_diag.loc[[weights_row > weight_threshold]]
            d2w_diag = d2w_diag.loc[[weights_row > weight_threshold]]
        b = np.matmul(designMatrix.transpose(),
                      (designMatrix * w_diag[:, None]))
        b_inv = np.linalg.inv(b)
        db = np.matmul(designMatrix.transpose(),
                       (designMatrix * dw_diag[:, None]))
        d2b = np.matmul(designMatrix.transpose(),
                        (designMatrix * d2w_diag[:, None]))
        detb = np.linalg.det(b)
        d_db = np.linalg.solve(b, db)
        ddetb = detb * trace(d_db)
        d2detb = detb * (
            pow(trace(d_db), 2)
            - trace(np.matmul(np.matmul(np.matmul(b_inv, db), b_inv), db))
            # - trace(np.matmul(d_db, d_db))
            + trace(np.linalg.solve(b, d2b))
        )
        cr_term = 0.5 * ((ddetb / detb) ** 2) - 0.5 * (d2detb / detb)
    else:
        cr_term = 0
    if use_weights:
        ll_part = -2 * (alpha ** -3) * (
            weights_row
            * (
                digamma(alpha_neg1)
                + np.log(1 + mu_row * alpha)
                - mu_row * alpha * pow((1 + mu_row * alpha), -1)
                - digamma(counts_row + alpha_neg1)
                + counts_row * pow((mu_row + alpha_neg1), -1)
            )
        ).sum() + alpha_neg2 * (
            weights_row
            * (
                -alpha_neg2 * trigamma(alpha_neg1)
                + pow(mu_row, 2) * alpha * pow((1 + mu_row * alpha), -2)
                + alpha_neg2 * trigamma(counts_row + alpha_neg1)
                + alpha_neg2 * counts_row * pow((mu_row + alpha_neg1), -2)
            )
        ).sum()
    else:
        ll_part = -2 * (alpha ** -3) * (
            (
                digamma(alpha_neg1)
                + np.log(1 + mu_row * alpha)
                - mu_row * alpha * pow((1 + mu_row * alpha), -1)
                - digamma(counts_row + alpha_neg1)
                + counts_row * pow((mu_row + alpha_neg1), -1)
            )
        ).sum() + alpha_neg2 * ((
                -alpha_neg2 * trigamma(alpha_neg1)
                + pow(mu_row, 2) * alpha * pow((1 + mu_row * alpha), -2)
                + alpha_neg2 * trigamma(counts_row + alpha_neg1)
                + alpha_neg2 * counts_row * pow((mu_row + alpha_neg1), -2)
            )
        ).sum()
    if use_prior:
        prior_part = -1 / log_alpha_prior_sigmasq
    else:
        prior_part = 0
    lp = (
        (ll_part + cr_term) * (alpha ** 2)
        + dlog_posterior(
            log_alpha,
            counts_row,
            mu_row,
            designMatrix_original,
            log_alpha_prior_mean,
            log_alpha_prior_sigmasq,
            False,
            weights_row,
            use_weights,
            weight_threshold,
            use_CR,
        )
        + prior_part
    )
    return lp


def estimateDispersion(
    np.ndarray[np.int64_t, ndim=2, mode="c"] counts,
    designMatrix,
    np.ndarray[np.float64_t, ndim=2, mode="c"] mu,
    double[::1] log_alphas,
    double[::1] log_alpha_prior_means,
    double log_alpha_prior_sigmasq,
    float minLogAlpha,
    double kappa0,
    float tolerance,
    int maxit,
    bint use_prior,
    np.ndarray[np.float64_t, ndim=2, mode="c"] weights,
    bint use_weights,
    float weight_threshold,
    bint use_CR,
    bint compute_d2log_posterior
):
    cdef:
        double epsilon = 1e-4
        int n_rows = len(counts)  # counts.shape[0]

        double[::1] new_log_alphas = np.empty(n_rows, dtype=np.double, order="C")
        double[::1] initial_lp = np.empty(n_rows, dtype=np.double, order="C")
        double[::1] initial_dlp = np.empty(n_rows, dtype=np.double, order="C")
        double[::1] last_lp = np.empty(n_rows, dtype=np.double, order="C")
        double[::1] last_dlp = np.empty(n_rows, dtype=np.double, order="C")
        double[::1] last_d2lp = np.empty(n_rows, dtype=np.double, order="C")
        double[::1] last_change = np.empty(n_rows, dtype=np.double, order="C")
        int[::1] iter = np.zeros(n_rows, dtype=np.int32, order="C")
        int[::1] iter_accept = np.zeros(n_rows, dtype=np.int32, order="C")

        Py_ssize_t row = 0
        double log_alpha
        double log_alpha_prior_mean
        double lp
        double lp_new
        double dlp
        double kappa
        double change
        double proposed_alpha
        double theta_kappa
        double theta_hat_kappa

        np.ndarray[np.int64_t, ndim= 1, mode = "c"] counts_row
        np.ndarray[np.float64_t, ndim= 1, mode = "c"] mu_row
        np.ndarray[np.float64_t, ndim= 1, mode = "c"] weights_row

    for row in xrange(n_rows):
        counts_row = counts[row]
        mu_row = mu[row]
        weights_row = np.array([]) if weights is None else weights[row]
        log_alpha = log_alphas[row]
        log_alpha_prior_mean = log_alpha_prior_means[row]

        lp = log_posterior(
            log_alpha,
            counts_row,
            mu_row,
            designMatrix,
            log_alpha_prior_mean,
            log_alpha_prior_sigmasq,
            use_prior,
            weights_row,
            use_weights,
            weight_threshold,
            use_CR,
        )
        dlp = dlog_posterior(
            log_alpha,
            counts_row,
            mu_row,
            designMatrix,
            log_alpha_prior_mean,
            log_alpha_prior_sigmasq,
            use_prior,
            weights_row,
            use_weights,
            weight_threshold,
            use_CR,
        )

        kappa = kappa0
        initial_lp[row] = lp
        initial_dlp[row] = dlp

        change = -1.0
        last_change[row] = change
        for i in xrange(maxit):
            iter[row] += 1
            proposed_alpha = log_alpha + kappa * dlp

            if proposed_alpha < -30:
                kappa = (-30 - log_alpha) / dlp
            if proposed_alpha > 10:
                kappa = (10 - log_alpha) / dlp

            theta_kappa = -log_posterior(
                proposed_alpha,
                counts_row,
                mu_row,
                designMatrix,
                log_alpha_prior_mean,
                log_alpha_prior_sigmasq,
                use_prior,
                weights_row,
                use_weights,
                weight_threshold,
                use_CR,
            )
            theta_hat_kappa = -lp - kappa * epsilon * (dlp ** 2)

            if theta_kappa <= theta_hat_kappa:
                iter_accept[row] += 1
                log_alpha = log_alpha + kappa * dlp
                lp_new = log_posterior(
                    log_alpha,
                    counts_row,
                    mu_row,
                    designMatrix,
                    log_alpha_prior_mean,
                    log_alpha_prior_sigmasq,
                    use_prior,
                    weights_row,
                    use_weights,
                    weight_threshold,
                    use_CR,
                )

                change = lp_new - lp
                if change < tolerance:
                    lp = lp_new
                    break
                if log_alpha < minLogAlpha:
                    break

                lp = lp_new
                dlp = dlog_posterior(
                    log_alpha,
                    counts_row,
                    mu_row,
                    designMatrix,
                    log_alpha_prior_mean,
                    log_alpha_prior_sigmasq,
                    use_prior,
                    weights_row,
                    use_weights,
                    weight_threshold,
                    use_CR,
                )
                kappa = min(kappa * 1.1, kappa0)

                if iter_accept[row] % 5 == 0:
                    kappa = kappa / 2
            else:
                kappa = kappa / 2

        last_lp[row] = lp
        last_dlp[row] = dlp
        if compute_d2log_posterior:
            last_d2lp[row] = d2log_posterior(
                log_alpha,
                counts_row,
                mu_row,
                designMatrix,
                log_alpha_prior_mean,
                log_alpha_prior_sigmasq,
                use_prior,
                weights_row,
                use_weights,
                weight_threshold,
                use_CR,
            )
        new_log_alphas[row] = log_alpha
        last_change[row] = change

    results = {
            "log_alphas": np.asarray(new_log_alphas),
            "iter": np.asarray(iter),
            "iter_accept": np.asarray(iter_accept),
            "last_change": np.asarray(last_change),
            "initial_lp": np.asarray(initial_lp),
            "initial_dlp": np.asarray(initial_dlp),
            "last_lp": np.asarray(last_lp),
            "last_dlp": np.asarray(last_dlp),
        }
    if compute_d2log_posterior:
        results.update({"last_d2lp": np.asarray(last_d2lp)})

    return results


def estimateDispersionGrid(
    np.ndarray[np.int64_t, ndim=2, mode="c"] counts,
    designMatrix,
    np.ndarray[np.float64_t, ndim=2, mode="c"] mu,
    double[::1] log_alpha_prior_means,
    double log_alpha_prior_sigmasq,
    bint use_prior,
    np.ndarray[np.float64_t, ndim=2, mode="c"] weights,
    bint use_weights,
    float weight_threshold,
    bint use_CR,
):
    cdef:
        n_rows = len(counts)  # counts.shape[0]
        Py_ssize_t row = 0
        int grid_size = 20
        float minLogAlpha = log(1e-8)
        # counts.shape[1]
        float max_log_alpha = log(max(10, len(designMatrix)))
        double log_alpha
        double log_alpha_prior_mean
        Py_ssize_t maxlp_idx
        double log_alpha_hat

        double[::1] lp = np.empty(grid_size, dtype=np.double, order="C")
        double[::1] log_alphas = np.empty(n_rows, dtype=np.double, order="C")
        double[::1] disp_grid = np.linspace(start=minLogAlpha,
                                            stop=max_log_alpha, num=grid_size).astype(np.double)
        float delta = disp_grid[1] - disp_grid[0]
        double[::1] disp_grid_fine

        np.ndarray[np.int64_t, ndim = 1, mode = "c"] counts_row
        np.ndarray[np.float64_t, ndim = 1, mode = "c"] mu_row
        np.ndarray[np.float64_t, ndim = 1, mode = "c"] weights_row

    for row in xrange(n_rows):
        counts_row = counts[row]
        mu_row = mu[row]
        weights_row = np.array([]) if weights is None else weights[row]
        log_alpha_prior_mean = log_alpha_prior_means[row]

        for i in xrange(grid_size):
            log_alpha = disp_grid[i]

            lp[i] = log_posterior(
                log_alpha,
                counts_row,
                mu_row,
                designMatrix,
                log_alpha_prior_mean,
                log_alpha_prior_sigmasq,
                use_prior,
                weights_row,
                use_weights,
                weight_threshold,
                use_CR,
            )

        maxlp_idx = np.argmax(lp)
        log_alpha_hat = disp_grid[maxlp_idx]
        disp_grid_fine = np.linspace(
            start=log_alpha_hat - delta, stop=log_alpha_hat + delta, num=grid_size
        )

        for i in xrange(grid_size):
            log_alpha = disp_grid_fine[i]

            lp[i] = log_posterior(
                log_alpha,
                counts_row,
                mu_row,
                designMatrix,
                log_alpha_prior_mean,
                log_alpha_prior_sigmasq,
                use_prior,
                weights_row,
                use_weights,
                weight_threshold,
                use_CR,
            )

        maxlp_idx = np.argmax(lp)
        log_alphas[row] = disp_grid_fine[maxlp_idx]

        return np.asarray(log_alphas)
