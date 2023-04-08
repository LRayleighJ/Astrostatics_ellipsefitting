# Astrostatics_ellipsefitting
NAOC astrostatics very_BIG_homework ex.2

Add the code of Fisher Matrix Forecast and modify the form of sersic profile. Just open pipeline_modified_sersic.py and HAVE FUN. About Fisher matrix, I add a quick guide. And please check [Report of the DARK ENERGY TASK FORCE](https://arxiv.org/abs/astro-ph/0609591), Page 94. LRayleighJ 230408

## 目前的进度和预期需要完成的事情

后面可以做的事情：
* 数据的进一步精细化分析和处理。SDSS DR16给出的数据下载界面有5栏数据，需要报告清楚它们的作用（可能用不到但是至少要清楚），确认数据的读取方法是可靠的。其实这是一个很重要的问题（我并没有去确认这一件事情。所以可能有很大的坑）
* 目前的图像里面有对椭圆拟合影响比较大的其他星体，有没有可能尽量降低这个影响
* 检验目前的code有无问题
* 处理不同波段和不同星系，分别给出其相应结果。
* 其他可能的拟合方法。`photutils`进行椭圆拟合采用的是傅里叶变换(maybe)，我之前写的那一坨trash思路是找焦点。应该还有更多的拟合方法，可以多搞几个测试一下有效性
* 采用多种方法进行误差分析并进行相互验证。目前程序使用mcmc对`lmfit`自动给出的误差进行了检验。可以有更多的统计方法做这件事情。（我目前想到的一个是可以用Fisher Matrix做不确定度分析，当然还有更多）
* 整理书面报告，slides和课堂pre
* ……
想到什么了之后可以再加，包括这一套code，anyone有什么想实现的功能可以随时修改、添加。

现在的code可以干什么：
* 数据的读取（直接把correct frame当成了最后的“完美数据”使用）
* 椭圆几何参数拟合（`photutils`真好用）。
* 拟合sersic函数，并给出置信椭圆进行误差分析
* MCMC验证置信椭圆的可靠性

学长写的东西真好使（心虚
## Develop Log

[OUTDATE] Add the code of Fisher Matrix Forecast. Just open pipeline_v2.py and HAVE FUN. About Fisher matrix, please check [Report of the DARK ENERGY TASK FORCE](https://arxiv.org/abs/astro-ph/0609591), Page 94. LRayleighJ 230407

[OUTDATE] Just open pipeline_v2.py and HAVE FUN. tutorial.ipynb and pipeline.py are f*cking TRASH.  LRayleighJ 230402

[OUTDATE] Just open tutorial.ipynb and HAVE FUN. LRayleighJ 230401

