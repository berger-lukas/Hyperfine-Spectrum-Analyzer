import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from spectrum_intensity_v7 import normalize_catalog, scale_figure, normalized_relayout


class IntensityTests(unittest.TestCase):
    def test_reference_excludes_stronger_outside_band(self):
        frame = pd.DataFrame({'Freq':[4000,5000,12000,20000,50000], 'Intensity':[400,2,4,8,1000]})
        original = frame['Intensity'].copy()
        self.assertEqual(normalize_catalog(frame,5000,20000),8)
        self.assertEqual(frame.loc[3,'Norm_Intensity'],1)
        self.assertEqual(frame.loc[2,'Norm_Intensity'],.5)
        pd.testing.assert_series_equal(original,frame['Intensity'])

    def test_no_overlap_does_not_use_remote_peak(self):
        frame = pd.DataFrame({'Freq':[30000,50000], 'Intensity':[3,20]})
        self.assertIsNone(normalize_catalog(frame,5000,20000))
        self.assertTrue((frame['Norm_Intensity']==0).all())

    def test_scaling_includes_fit_traces_and_preserves_domain_shapes(self):
        fig = go.Figure([go.Scatter(y=[0,.5,1]),go.Scatter(y=[0,-1,None]),go.Scatter(y=[.2,.6])])
        fig.update_yaxes(range=[-.1,1.1])
        fig.add_shape(type='rect',yref='y domain',x0=1,x1=2,y0=0,y1=1)
        fig.add_shape(type='line',yref='y',x0=1,x1=1,y0=0,y1=.8)
        scale_figure(fig,250,'Intensity (mV)')
        np.testing.assert_allclose(fig.data[0].y,[0,125,250])
        self.assertEqual(fig.data[1].y[1],-250)
        self.assertEqual(fig.layout.shapes[0].y1,1)
        self.assertEqual(fig.layout.shapes[1].y1,200)
        self.assertEqual(fig.layout.yaxis.title.text,'Intensity (mV)')
        self.assertEqual(normalized_relayout({'yaxis.range':list(fig.layout.yaxis.range),'xaxis.range':[5,20]},250),
                         {'yaxis.range':[-.1,1.1],'xaxis.range':[5,20]})


if __name__ == '__main__':
    unittest.main(verbosity=2)
