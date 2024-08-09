import numpy as np
import unittest
import matplotlib.pyplot as plt
from unittest import mock
from driftbench.data_generation.latent_information import LatentInformation
import driftbench.data_generation.visualize as visualize


class TestPlotCurveWithLatentInformation(unittest.TestCase):
    def setUp(self):
        self.p = lambda w, x: w[0] * x ** 3 + w[1] * x ** 2 + w[2] * x + w[3]
        x0 = np.array([0., 2., 4.])
        y0 = np.array([0., 8., 64.])
        x1 = np.array([1., 3.])
        y1 = np.array([3., 27.])
        x2 = np.array([2.])
        y2 = np.array([12.])
        self.latent_information = LatentInformation(y0, x0, y1, x1, y2, x2)

    @mock.patch("%s.visualize.plt" % __name__)
    def test_plot_curve_with_latent_information_with_default_params(self, mock_plt):
        w = np.array([1., 0., 0., 0.])
        mock_plt.subplots.return_value = (mock.MagicMock(), mock.MagicMock())
        mock_fig, mock_ax = mock_plt.subplots()
        visualize.plot_curve_with_latent_information(w, self.p, self.latent_information)
        mock_ax.plot.assert_called()
        mock_ax.axvline.assert_any_call(0., linestyle="dashed", color="black")
        mock_ax.axvline.assert_any_call(2., linestyle="dashed", color="black")
        mock_ax.axvline.assert_any_call(4., linestyle="dashed", color="black")
        mock_ax.axvline.assert_called_with(2., linestyle="dashed", color="purple", label="convex")
        mock_ax.scatter.assert_any_call(0., 0., color="red")
        mock_ax.scatter.assert_any_call(2., 8., color="red")
        mock_ax.scatter.assert_any_call(4., 64., color="red")
        mock_ax.legend.assert_called_once()

    @mock.patch("%s.visualize.plt" % __name__)
    def test_plot_with_custom_title(self, mock_plt):
        w = np.array([1., 0., 0., 0.])
        mock_plt.subplots.return_value = (mock.MagicMock(), mock.MagicMock())
        mock_fig, mock_ax = mock_plt.subplots()
        title = "Curve fitting for p(w, x) = x^3"
        visualize.plot_curve_with_latent_information(w, self.p, self.latent_information, title=title)
        mock_ax.set_title.assert_called_once_with(title)

    @mock.patch("%s.visualize.plt" % __name__)
    def test_plot_with_extern_ax(self, mock_plt):
        w = np.array([1., 0., 0., 0.])
        fig, ax = plt.subplots()
        visualize.plot_curve_with_latent_information(w, self.p, self.latent_information,
                                                     ax=ax)
        mock_plt.subplots.assert_not_called()

    @mock.patch("%s.visualize.plt" % __name__)
    def test_plot_with_ylim(self, mock_plt):
        w = np.array([1., 0., 0., 0.])
        mock_plt.subplots.return_value = (mock.MagicMock(), mock.MagicMock())
        mock_fig, mock_ax = mock_plt.subplots()
        y_lim = (-5, 5)
        visualize.plot_curve_with_latent_information(w, self.p, self.latent_information,
                                                     y_lim=y_lim)
        mock_ax.set_ylim.assert_called_once_with(y_lim)


class TestPlotCurves(unittest.TestCase):
    def setUp(self):
        self.curves = np.arange(2000).reshape((200, 10))

    @mock.patch("%s.visualize.plt" % __name__)
    def test_plot_curves_with_default_params(self, mock_plt):
        mock_plt.subplots.return_value = (mock.MagicMock(), mock.MagicMock())
        mock_fig, mock_ax = mock_plt.subplots()
        xs = np.linspace(0, 100, self.curves.shape[1])
        visualize.plot_curves(self.curves, xs)
        mock_plt.get_cmap.assert_called_once_with(name="coolwarm")
        mock_ax.plot.assert_called()
        mock_ax.set_title.assert_not_called()
        mock_ax.set_ylim.assert_not_called()

    @mock.patch("%s.visualize.plt" % __name__)
    def test_plot_curves_with_custom_title(self, mock_plt):
        mock_plt.subplots.return_value = (mock.MagicMock(), mock.MagicMock())
        mock_fig, mock_ax = mock_plt.subplots()
        title = "Curves plot"
        xs = np.linspace(0, 100, self.curves.shape[1])
        visualize.plot_curves(xs, self.curves, title=title)
        mock_ax.set_title.assert_called_once_with(title)

    @mock.patch("%s.visualize.plt" % __name__)
    def test_plot_curves_with_custom_cmap(self, mock_plt):
        mock_plt.subplots.return_value = (mock.MagicMock(), mock.MagicMock())
        cmap = "BrBG"
        xs = np.linspace(0, 100, self.curves.shape[1])
        visualize.plot_curves(xs, self.curves, cmap=cmap)
        mock_plt.get_cmap.assert_called_once_with(name=cmap)

    @mock.patch("%s.visualize.plt" % __name__)
    def test_plot_curves_with_custom_ylim(self, mock_plt):
        mock_plt.subplots.return_value = (mock.MagicMock(), mock.MagicMock())
        mock_fig, mock_ax = mock_plt.subplots()
        ylim = np.quantile(self.curves, q=0.02), np.quantile(self.curves, q=0.98)
        xs = np.linspace(0, 100, self.curves.shape[1])
        visualize.plot_curves(xs, self.curves, ylim=ylim)
        mock_ax.set_ylim.assert_called_once_with(ylim)
