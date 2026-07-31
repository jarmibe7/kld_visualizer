# KL Divergence Visualizer

This project is a small static web app that compares how a single Gaussian approximation Q fits a fixed target mixture distribution P under two different objectives:

- Reverse KL: $D_{KL}(Q \| P)$, which is more mode-seeking.
- Forward KL: $D_{KL}(P \| Q)$, which is more mass-covering.

The page uses plain HTML, CSS, and JavaScript with no build step. It is designed to be published directly to GitHub Pages from the repository root.

## What to expect

- Adjust the target mixture components to change the shape of P.
- Watch Q evolve through gradient descent for both KL directions side by side.
- Compare how reverse KL collapses onto a dominant mode while forward KL spreads to cover the full target support.

## Background

- [KL divergence on Wikipedia](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
- [Variational inference overview](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)

## GitHub Pages

When the repository is published with GitHub Pages enabled for the main branch, the site is available at:

https://jarmibe7.github.io/kld_visualizer/
