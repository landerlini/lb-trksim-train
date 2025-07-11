import numpy as np 
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm 
from logging import getLogger as logger

################################################################################
# Access to data
def getData(cfg, chunk, tracktype):
    eps = cfg.outlayerBeyondThreshold/2
    chunk.query(" and ".join(["(%s)" % s for s in cfg.cuts]), inplace=True)
    chunk.query("type==%d" % tracktype, inplace=True)
    X = np.stack([chunk.eval(v).astype(np.float64) for v in cfg.discrVars]).T
    n = X.shape[0]
    X *= np.stack([np.random.choice([-1., 1.], n)
                   if v in cfg.symVars else np.ones(n) for v in cfg.discrVars]).T
    X += np.stack([np.random.uniform(0, 1, n)
                   if v in cfg.intVars else np.zeros(n) for v in cfg.discrVars]).T
    Y = np.stack([chunk.eval(v).astype(np.float64) for v in cfg.targetVars]).T
    Y *= np.stack([np.random.choice([-1., 1.], n)
                   if v in cfg.symVars else np.ones(n) for v in cfg.targetVars]).T
    Y += np.stack([np.random.uniform(0, 1, n)
                   if v in cfg.intVars else np.zeros(n) for v in cfg.targetVars]).T
    mask = np.ones(len(X), dtype=np.bool)
    # filters outlayers, for non-boolean variables
    for name, x in zip(cfg.discrVars, X.T):
        if ("==" not in name) and (">" not in name) and ("<" not in name):
            mask &= (x >= np.quantile(x, eps)) & (x <= np.quantile(x, 1-eps))
    for name, y in zip(cfg.targetVars, Y.T):
        if ("==" not in name) and (">" not in name) and ("<" not in name):
            mask &= (y >= np.quantile(y, eps)) & (y <= np.quantile(y, 1-eps))
    X = X[mask]
    Y = Y[mask]
    return X, Y


################################################################################
# Agreement monitoring
def computeLoss(X, Y, Yhat):
    "Trains a BDT to distinguish X:Yhat from X:Y and returns accuracy (train,test)"
    classifier = GradientBoostingClassifier(n_estimators=10, max_depth=2)
    N = len(X)//2
    tXY = np.concatenate([X[:N], Y[:N]], axis=1)[:N]
    tXYhat = np.concatenate([X[N:], Yhat[N:]], axis=1)[:N]
    vXY = np.concatenate([X[N:], Y[N:]], axis=1)[:N]
    vXYhat = np.concatenate([X[:N], Yhat[:N]], axis=1)[:N]
    classifier.fit(
        np.concatenate([tXY, tXYhat], axis=0),
        np.concatenate([np.ones(N), np.zeros(N)], axis=0)
    )

    tset = np.concatenate([tXY, tXYhat], axis=0)
    tlab = np.concatenate([np.ones(N), np.zeros(N)], axis=0)
    tpred = classifier.predict_proba(tset)[:, 1]

    vset = np.concatenate([vXY, vXYhat], axis=0)
    vlab = np.concatenate([np.ones(N), np.zeros(N)], axis=0)
    vpred = classifier.predict_proba(vset)[:, 1]


    return (roc_auc_score(tlab, tpred), roc_auc_score(vlab, vpred))


################################################################################
## Training loop 
def train(cfg, gan, dataset, tracktype):
    losses = []
    bdt_losses = []
    progress_bar = tqdm(enumerate(dataset),
                        total=cfg.nChunks,
                        ascii=True,
                        desc="Starting..."
                        )

    for iChunk, chunk in progress_bar:
        if iChunk >= cfg.nChunks:
            break

        X, Y = getData(cfg, chunk, tracktype)
        gan.fit(X, Y)
        losses += gan.losses_

        Yhat = gan.predict(X[:10000])

        bdt_t, bdt_v = computeLoss(
            gan.transformerX_.transform(X[:10000]),
            gan.transformerY_.transform(Y[:10000]),
            gan.transformerY_.transform(Yhat[:10000])
        )

        bdt_losses.append([len(losses), bdt_t, bdt_v])
        progress_bar.set_description("Loss: %.1f%%" % (bdt_v*100.))

        logger("gan_utils.train").info("""--------------
    Status after training on chunk:     %(iChunk)d/%(totChunks)d
    Loss GAN:           %(ganloss).3f
    Loss BDT (test):    %(tloss).3f
    Loss BDT (train):   %(vloss).3f
  """ % dict(
            iChunk=iChunk + 1,
            totChunks=cfg.nChunks,
            tloss=bdt_t,
            vloss=bdt_v,
            ganloss=np.mean(losses[-100:])
        ))

        yield np.array(losses).reshape(-1), np.array(bdt_losses)

