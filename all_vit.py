import timm
vits = timm.list_models(module="vision_transformer", pretrained=True)
for vit in vits:
    print(vit)