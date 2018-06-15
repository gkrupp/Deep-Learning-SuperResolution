## Futtatás:
`sudo python3 <netname>.py -predindex <imgindex1 imgindex2 ...>`
**Adathalmaz 0., 1., 2. képén futtatva as `srcnn` hálót:**
`sudo python3 srcnn.py -predindex 0 1 2`

## Tanítás:
`sudo python3 <netname>.py -train [-n train_set_size] [-v validation_set_size] [-b batch_size] [-data dataset_name] [-model model_name] [-chkimg <imgindex1 imgindex2 ...>] [-imgpath image_saving_path]`
**`srcnn` tantása és a model mentése `models/srcnn_2.h5` fájlba, 20000 képen 1000 validációs képpel és 200-as batch mérettel, epoch-onként kimentve a 11. részeredmény képet:**
`sudo python3 srcnn.py -train -model "models/srcnn_2.h5" -chkimg 11 -n 20000 -v 1000 -b 200`

## Megjegyzés:
Előre tanított modellek:
`models/*` alapértelmezésben `srcnn.py -> models/scrnn.h5`


## Adathalmaz:
A teljes adathalmaz 120804 képből áll, mérete miatt (13GB) a csatolt csak 40000 képet tertalmaz.
Az adatfájl 3 adatmezőt tartemaz:
* `32x32`: 32x32-es downsapled képek `float32(40000, 32, 32, 3)`
* `64x64`: 64x64-es downsampled képek `float32(40000, 64, 64, 3)`
* `64x64lanczos`: 32x32-ből 64x64-esre nagyított képek _lanczos_ interpolációval `float32(40000, 64, 64, 3)`
* `order`: eredeti fájlnevek sorrendben `|S21(40000,)`

A képek pixeleinek intenzitása `[0,1]` intervallumra normált.

