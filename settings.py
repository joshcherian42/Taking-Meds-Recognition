global sampling_rate
sampling_rate = 70
global labels
labels = ["Start", "End", "Origin", "Activity", "pred"]
global activities
activities = ["nothing", "taking meds"]
global _acts
_acts = ["taking meds"]

global cv
cv = 10

global out_cols
out_cols = labels + activities
out_cols += ["pred_" + str(j) for j in range(cv)]

for act in activities:
    out_cols += [act + "_" + str(j) for j in range(cv)]


global features_header
features_header = ['Avg Jerk Acc Right X',
                   'Avg Jerk Acc Right Y',
                   'Avg Jerk Acc Right Z',
                   'Avg Height Acc Right X',
                   'Avg Height Acc Right Y',
                   'Avg Height Acc Right Z',
                   'Stdev Height Acc Right X',
                   'Stdev Height Acc Right Y',
                   'Stdev Height Acc Right Z',
                   'Energy Acc Right X',
                   'Energy Acc Right Y',
                   'Energy Acc Right Z',
                   'Entropy Acc Right X',
                   'Entropy Acc Right Y',
                   'Entropy Acc Right Z',
                   'Average Acc Right X',
                   'Average Acc Right Y',
                   'Average Acc Right Z',
                   'Standard Deviation Acc Right X',
                   'Standard Deviation Acc Right Y',
                   'Standard Deviation Acc Right Z',
                   'RMS Acc Right X',
                   'RMS Acc Right Y',
                   'RMS Acc Right Z',
                   'Num Peaks Acc Right X',
                   'Num Peaks Acc Right Y',
                   'Num Peaks Acc Right Z',
                   'Average Peaks Acc Right X',
                   'Average Peaks Acc Right Y',
                   'Average Peaks Acc Right Z',
                   'Standard Deviation Peaks Acc Right X',
                   'Standard Deviation Peaks Acc Right Y',
                   'Standard Deviation Peaks Acc Right Z',
                   'Num Valleys Acc Right X',
                   'Num Valleys Acc Right Y',
                   'Num Valleys Acc Right Z',
                   'Average Valleys Acc Right X',
                   'Average Valleys Acc Right Y',
                   'Average Valleys Acc Right Z',
                   'Standard Deviation Valleys Acc Right X',
                   'Standard Deviation Valleys Acc Right Y',
                   'Standard Deviation Valleys Acc Right Z',
                   'Axis Overlap Acc Right',
                   'Fractal Dimension Acc Right',
                   'Spectral Centroid Acc Right X',
                   'Spectral Centroid Acc Right Y',
                   'Spectral Centroid Acc Right Z',
                   'Spectral Spread Acc Right X',
                   'Spectral Spread Acc Right Y',
                   'Spectral Spread Acc Right Z',
                   'Spectral Rolloff Acc Right X',
                   'Spectral Rolloff Acc Right Y',
                   'Spectral Rolloff Acc Right Z',
                   'Avg Jerk Acc Left X',
                   'Avg Jerk Acc Left Y',
                   'Avg Jerk Acc Left Z',
                   'Avg Height Acc Left X',
                   'Avg Height Acc Left Y',
                   'Avg Height Acc Left Z',
                   'Stdev Height Acc Left X',
                   'Stdev Height Acc Left Y',
                   'Stdev Height Acc Left Z',
                   'Energy Acc Left X',
                   'Energy Acc Left Y',
                   'Energy Acc Left Z',
                   'Entropy Acc Left X',
                   'Entropy Acc Left Y',
                   'Entropy Acc Left Z',
                   'Average Acc Left X',
                   'Average Acc Left Y',
                   'Average Acc Left Z',
                   'Standard Deviation Acc Left X',
                   'Standard Deviation Acc Left Y',
                   'Standard Deviation Acc Left Z',
                   'RMS Acc Left X',
                   'RMS Acc Left Y',
                   'RMS Acc Left Z',
                   'Num Peaks Acc Left X',
                   'Num Peaks Acc Left Y',
                   'Num Peaks Acc Left Z',
                   'Average Peaks Acc Left X',
                   'Average Peaks Acc Left Y',
                   'Average Peaks Acc Left Z',
                   'Standard Deviation Peaks Acc Left X',
                   'Standard Deviation Peaks Acc Left Y',
                   'Standard Deviation Peaks Acc Left Z',
                   'Num Valleys Acc Left X',
                   'Num Valleys Acc Left Y',
                   'Num Valleys Acc Left Z',
                   'Average Valleys Acc Left X',
                   'Average Valleys Acc Left Y',
                   'Average Valleys Acc Left Z',
                   'Standard Deviation Valleys Acc Left X',
                   'Standard Deviation Valleys Acc Left Y',
                   'Standard Deviation Valleys Acc Left Z',
                   'Axis Overlap Acc Left',
                   'Fractal Dimension Acc Left',
                   'Spectral Centroid Acc Left X',
                   'Spectral Centroid Acc Left Y',
                   'Spectral Centroid Acc Left Z',
                   'Spectral Spread Acc Left X',
                   'Spectral Spread Acc Left Y',
                   'Spectral Spread Acc Left Z',
                   'Spectral Rolloff Acc Left X',
                   'Spectral Rolloff Acc Left Y',
                   'Spectral Rolloff Acc Left Z',
                   'Activity', 'Start', 'End']
