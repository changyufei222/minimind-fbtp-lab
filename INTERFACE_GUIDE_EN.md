# MiniMind FBTP Lab Interface Guide

[English](./INTERFACE_GUIDE_EN.md) | [中文](./INTERFACE_GUIDE_CN.md)

## Purpose

Small-model Query Compiler experiment with DSL, validator, and holdout gate.

## Intended Readers

Model-training reviewers and interviewers checking small-model adaptation rather than a from-scratch MiniMind rewrite.

## How To Read This Repository

- Start from README.md, TRAINING_README.md, UPSTREAM.md, and FINAL_RESULT_SUMMARY.md.
- query_compiler/ contains DSL, validator, repair, scoring, and executor logic.
- reports/algorithm_resume/ contains resume-facing evidence.
- No model weights or private raw training data are uploaded.

## Repository Boundary

This repository is an upload-ready public package. Local paths, runtime caches, logs, private raw data, model weights, and temporary working files were excluded before upload.

## Language Switch

Use the links at the top of this file to switch between the English and Chinese interface guides.