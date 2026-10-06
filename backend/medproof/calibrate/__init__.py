"""Calibration for the specialist readers: temperature scaling, conformal sets, abstention and selective prediction.

Pure numpy / scipy, no torch, so the API can load a fitted calibrator without the training stack.
Fitting happens offline in ml/ on held-out validation and calibration splits; serving only applies it.
"""
