import argparse
import sensors
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

sensors.init()

pmbus_ultra96v2 = {
    # ir38060-i2c-6-45
    "ir38060-i2c-6-45:iout1" : "5V",
    # irps5401-i2c-6-43
    "irps5401-i2c-6-43:iout1" : "VCCAUX",
    "irps5401-i2c-6-43:iout2" : "VCCO 1.2V",
    "irps5401-i2c-6-43:iout3" : "VCCO 1.1V",
    "irps5401-i2c-6-43:iout4" : "VCCINT",
    "irps5401-i2c-6-43:iout5" : "3.3V DP",
    # irps5401-i2c-6-44
    "irps5401-i2c-6-44:iout1" : "VCCPSAUX",
    "irps5401-i2c-6-44:iout2" : "PSINT_LP",
    "irps5401-i2c-6-44:iout3" : "VCCO 3.3V",
    "irps5401-i2c-6-44:iout4" : "PSINT_FP",
    "irps5401-i2c-6-44:iout5" : "PSPLL 1.2V",
}

pmbus_uz7ev_evcc = {
    # ir38063-i2c-3-4c
    "ir38063-i2c-3-4c:iout1" : "Carrier 3V3",
    # ir38063-i2c-3-4b
    "ir38063-i2c-3-4b:iout1" : "Carrier 1V8",
    # irps5401-i2c-3-4a
    "irps5401-i2c-3-4a:iout1" : "Carrier 0V9 MGTAVCC",
    "irps5401-i2c-3-4a:iout2" : "Carrier 1V2 MGTAVTT",
    "irps5401-i2c-3-4a:iout3" : "Carrier 1V1 HDMI",
    "irps5401-i2c-3-4a:iout4" : "Unused",
    "irps5401-i2c-3-4a:iout5" : "Carrier 1V8 MGTVCCAUX LDO",
    # irps5401-i2c-3-49
    "irps5401-i2c-3-49:iout1" : "Carrier 0V85 MGTRAVCC",
    "irps5401-i2c-3-49:iout2" : "Carrier 1V8 VCCO",
    "irps5401-i2c-3-49:iout3" : "Carrier 3V3 VCCO",
    "irps5401-i2c-3-49:iout4" : "Carrier 5V MAIN",
    "irps5401-i2c-3-49:iout5" : "Carrier 1V8 MGTRAVTT LDO",
    # ir38063-i2c-3-48
    "ir38063-i2c-3-48:iout1" : "SOM 0V85 VCCINT",
    # irps5401-i2c-3-47
    "irps5401-i2c-3-47:iout1" : "SOM 1V8 VCCAUX",
    "irps5401-i2c-3-47:iout2" : "SOM 3V3",
    "irps5401-i2c-3-47:iout3" : "SOM 0V9 VCUINT",
    "irps5401-i2c-3-47:iout4" : "SOM 1V2 VCCO_HP_66",
    "irps5401-i2c-3-47:iout5" : "SOM 1V8 PSDDR_PLL LDO",
    # irps5401-i2c-3-46
    "irps5401-i2c-3-46:iout1" : "SOM 1V2 VCCO_PSIO",
    "irps5401-i2c-3-46:iout2" : "SOM 0V85 VCC_PSINTLP",
    "irps5401-i2c-3-46:iout3" : "SOM 1V2 VCCO_PSDDR4_504",
    "irps5401-i2c-3-46:iout4" : "SOM 0V85 VCC_PSINTFP",
    "irps5401-i2c-3-46:iout5" : "SOM 1V2 VCC_PSPLL LDO",
}

pmbus_uz3eg_xxx = {
    # irps5401-i2c-3-43
    "irps5401-i2c-3-43:iout1" : "PSIO",
    "irps5401-i2c-3-43:iout2" : "VCCAUX",
    "irps5401-i2c-3-43:iout3" : "PSINTLP",
    "irps5401-i2c-3-43:iout4" : "PSINTFP",
    "irps5401-i2c-3-43:iout5" : "PSPLL",
    # irps5401-i2c-3-44
    "irps5401-i2c-3-44:iout1" : "PSDDR4",
    "irps5401-i2c-3-44:iout2" : "INT_IO",
    "irps5401-i2c-3-44:iout3" : "3.3V",
    "irps5401-i2c-3-44:iout4" : "INT",
    "irps5401-i2c-3-44:iout5" : "PSDDRPLL",
    # irps5401-i2c-3-45
    "irps5401-i2c-3-45:iout1" : "MGTAVCC",
    "irps5401-i2c-3-45:iout2" : "5V",
    "irps5401-i2c-3-45:iout3" : "3.3V",
    "irps5401-i2c-3-45:iout4" : "VCCO 1.8V",
    "irps5401-i2c-3-45:iout5" : "MGTAVTT",
}


pmbus_annotations = {
    "ULTRA96V2"   : pmbus_ultra96v2,
    "UZ7EV_EVCC"  : pmbus_uz7ev_evcc,
    "UZ3EG_IOCC"  : pmbus_uz3eg_xxx,
    "UZ3EG_PCIEC" : pmbus_uz3eg_xxx
}


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--target", required=False,
	help = "target hardware (ULTRA96V2, UZ7EV_EVCC, UZ3EG_IOCC, UZ3EG_PCIEC)")
args = vars(ap.parse_args())
#
use_annotations = 0
if not args.get("target",False):
    use_annotations = 0
    target = ""
else:
    target = args["target"]
    print('[INFO] target = ',target)
    if target in pmbus_annotations:
        target_annotations = pmbus_annotations[target]
        use_annotations = 1


pmbus_current_features = []
for chip in sensors.iter_detected_chips():
    device_name = str(chip)
    adapter_name = str(chip.adapter_name)
    print( "%s at %s" % (device_name, adapter_name) )
    #if 'irps5401' in device_name:
    #if 'ir38063' in device_name:
    if True:
        for feature in chip:
            feature_name = str(feature.label)
            if 'iout' in feature_name:
                label = device_name + ":" + feature_name
                pmbus_current_features.append( (label, feature) )
                feature_value = feature.get_value() * 1000.0
                print( " %s : %8.3f mA" % (feature_name, feature_value) )


N = len(pmbus_current_features)
c_x = np.linspace(1,N,N)
c_y = np.linspace(0,0,N,dtype=float)

c_l = []
for (i, (label,feature)) in enumerate(pmbus_current_features):
    c_l.append(label)
if use_annotations == 1:
    for i in range( len(c_l) ):
        label = c_l[i]
        if label in target_annotations:
            label = target_annotations[label]
        c_l[i] = label
    

fig = plt.figure()
#a = fig.add_subplot(1,1,1)
a = fig.add_subplot(1,2,2)
a.barh(c_x,c_y)

while True:

    for (i, (label,feature)) in enumerate(pmbus_current_features):
        c_y[i] = feature.get_value() * 1000.0

    fig.delaxes(a)
    #a = fig.add_subplot(1,1,1)
    a = fig.add_subplot(1,2,2)
    a.barh(c_x,c_y)
    a.set_xlabel('Current (mA)')
    a.set_xlim(xmin=0.0,xmax=10000.0)
    a.set_yticks(c_x)
    a.set_yticklabels(c_l)


    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    cv2.imshow("PMBUS(sensors) Power Metrics",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') :
        break

cv2.destroyAllWindows()

sensors.cleanup()
