from matplotlib import pyplot as plt

plt.style.use('dark_background')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#FFA500', '#FF4500', '#FF6347', '#FF8C00', '#FFD700'])
plt.rcParams['font.family'] = 'sans-serif'


def set_figure_pixel_size(width, height, ppi, f=None):
    if f:
        f.set_dpi(ppi)
        return f.set_size_inches(width / float(ppi), height / float(ppi))
    plt.figure(figsize=(width / float(ppi), height / float(ppi)), dpi=ppi)
    return None
