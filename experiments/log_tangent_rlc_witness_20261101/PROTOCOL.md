> 2026-11-08 精选归档：以下为原日期研究记录；“active／下一步／完成”等是当时状态。现行结论见[研究现状](../../docs/研究现状_20261108.md)。此目录是核心阅读与复核材料，原运行包未全量附带；JSON中的历史路径/证书不是本次重跑证明。

# 第二个开发信号：正RLC部分幅值过补偿的连续见证

首次计算前冻结。上一轮log-tangent第一个Lorentz开发信号的N128/256/512连续上下界仍重叠，保留未决。本轮按已冻结NEXT_CONTINUOUS_PROTOCOL取第二个双层信号partial72 index7，gain1 vs gain2，target_A9、shift−64，其余source/observer/完整指标不变。不称held-out，也不修改旧854文件封存。

主量仍rho=|F'/F|、mark=[2/pi atan(Re logF),2/pi Im logF,Re tau,Im tau]，source asinh(x/h)，Gaussian span6h，cone cost=|vt−vc|+min(vt,vc)(1−exp(−Euclidean distance of source+mark))，删除v，C=max(transport,原Gaussian复log均值锚)。本轮不使用服务器。

在正baseline RLC F=1+A/(1+iQ sinh(ln10(x−center)))引入t=asinh(Q sinh(ln10(x−center)))。质量rho dx=A/sqrt((1+A)^2+sinh(t)^2)dt，x(t)=center+asinh(sinh(t)/Q)/ln10。直接正元件网络、主kernel和此独立坐标必须桥接。

连续单位下界使用两个方向的运输dual，不需要解完整大LP：target势phi(x)=v(x)；candidate每个t-cell取常势psi_j=v_min,j−Bupper_j，其中Bupper_j是该candidate cell中对全部target x的benefit上界。benefit恰为min(vt,vc)(1+exp(−||p_t−p_c||))。两侧cell端点给出精确的Gaussian/rho最小最大；p在midpoint的球半径由导数上界×半cell宽。完整q导数范数≤sqrt(4+(2/pi)^2)：logF的t导数范数≤1，单位复切向角导数绝对值≤2。source导数局部上界为max_cell x'(t)/sqrt(h²+min_cell|x|²)，x'随|t|递增(Q>=1)。所以全p球半径可解析控制，不把局部scalar optimizer当全域最大。

target观察裁剪±12sigma外benefit≤2exp(−72)，并入全target supremum；candidate裁剪外取psi=−v，其负目标值由旧rho_x全局上界A ln10(Q+1)乘Gaussian尾控制。target外phi=v的正贡献可直接舍弃。cell质量用正rho_t单调包络，下界目标对psi正负分别选mass_lower/mass_upper；target正项使用rho_min*v_min*dt。这给全轴dual下界，不依赖求积收敛。浮点节点/端点以固定1e-10几何、1e-12可见度/密度余量向外扩展；最终实际使用分辨率的全部两侧midpoints及端点另以MP80/120核对，明确这些保护并非通用区间计算库证书。

固定下界t-cell分辨率1024/2048/4096/8192，按顺序计算，不调观察域/代价/删除。每层保存上下包络数组及dual值，并以独立实际t GK15/GK21验证visible mass与rho积分。gain2原初上界固定LP N128/256/512，在真实rho cell质量约束下构造product coupling，余密度删除。cell质量GL64/128/256与实际t GK15/GK21核对；flow按最小边缘比缩小1e-9。二维cost在y=x/y=2c_obs−x分片，GL8/16/32/64三阶双级1e-9+1e-8|ref|。主量均值锚由实际正网络复log GK15/GK21独立重算。

只要任一预定单位dual下界L与预定gain2可行原初上界U满足score(L)−score(U)>1e-8且>100倍独立参考差，并通过MP80/120全部字段/source/包络节点桥接，就确认本定义的连续正RLC反例并停止扩大。若仍重叠/求积未决/包络guard失败，保留未决，不改公式或gain追反例。解析对偶可行性和物理实现分别说明；积分参考不是任意精度区间包络。原所有40/440、20/167、完整Laurent/public640、4/8密度隔离、1.25全成本、60000同预算和MAG/PHA二维膝点均保留。
