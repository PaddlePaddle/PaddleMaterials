# Pretrained model package contract

Registered pretrained models must use a predictable archive layout. The archive
name, top-level directory, and configuration stem must all match the
`MODEL_REGISTRY` key:

```text
<model_name>.zip
└── <model_name>/
    ├── <model_name>.yaml
    └── checkpoints/
        └── best.pdparams
```

BCE link of weight should be recoded in ppmat/models/__init__.py `MODEL_REGISTRY`
