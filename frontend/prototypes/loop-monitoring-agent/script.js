const menus = [
  {
    key: "workbench",
    label: "工作台",
    eyebrow: "Workbench",
    pages: [
      ["overview", "总览驾驶舱", "装置级健康总览、异常回路、整定建议和本班任务入口。"],
      ["todos", "待处理回路", "把 Agent 发现的问题形成工程师待办队列。"],
      ["shift_tasks", "本班任务", "展示监控、诊断、整定、审批任务的运行状态。"],
      ["risk_alerts", "风险预警", "集中展示高风险回路和需要人工处理的风险。"],
    ],
  },
  {
    key: "monitoring",
    label: "回路监控",
    eyebrow: "Loop Monitoring",
    pages: [
      ["global_board", "全局回路看板", "按装置、回路类型和状态查看所有回路健康情况。"],
      ["loop_profile", "单回路画像", "查看单个回路的趋势、PID、阀门、报警和上下文。"],
      ["trend_spectrum", "趋势与频谱", "分析 PV/SV/MV 趋势、主频、滞后和相关性。"],
      ["alarm_events", "报警与事件", "把报警、操作、工况切换与控制性能关联起来。"],
    ],
  },
  {
    key: "assessment",
    label: "回路评估",
    eyebrow: "Loop Assessment",
    pages: [
      ["performance_score", "性能评分", "评价当前控制效果和执行机构负担。"],
      ["data_quality", "数据质量", "判断历史数据是否足够支撑分析和辨识。"],
      ["operating_condition", "工况识别", "识别稳态、负荷变化、开停车、扰动等工况。"],
      ["readiness", "整定准备度", "判断是否允许进入辨识和参数建议。"],
    ],
  },
  {
    key: "diagnostics",
    label: "根因诊断",
    eyebrow: "Diagnostics",
    pages: [
      ["diagnosis_overview", "诊断总览", "展示候选根因、置信度、证据链和建议动作。"],
      ["valve_diagnosis", "阀门诊断", "识别饱和、卡涩、死区、回差、频繁反向。"],
      ["oscillation_diagnosis", "振荡诊断", "区分自激振荡、外扰振荡、多回路耦合。"],
      ["model_reliability", "模型可靠性", "解释辨识模型能否用于整定。"],
    ],
  },
  {
    key: "tuning",
    label: "整定中心",
    eyebrow: "Tuning Center",
    pages: [
      ["tuning_tasks", "整定任务", "管理 Agent 自动触发和人工发起的整定任务。"],
      ["identification", "窗口与辨识", "展示候选窗口、模型池、多轮 attempts 和 LLM 评审。"],
      ["pid_candidates", "参数候选", "比较 IMC、Lambda、ZN、CHR 等整定策略。"],
      ["approval", "下发确认", "展示新旧参数对比、风险确认、审批和回滚方案。"],
    ],
  },
  {
    key: "knowledge",
    label: "经验中心",
    eyebrow: "Knowledge Center",
    pages: [
      ["cases", "整定案例库", "沉淀诊断、整定、下发和效果回看案例。"],
      ["rules", "规则库", "维护回路先验、参数约束和评分规则。"],
      ["knowledge_graph", "知识图谱", "维护装置、设备、回路、位号和上下游关系。"],
      ["model_versions", "模型版本", "管理辨识模型、评分规则和 Agent 策略版本。"],
    ],
  },
  {
    key: "settings",
    label: "系统设置",
    eyebrow: "System Settings",
    pages: [
      ["data_sources", "数据源配置", "配置时序库、DCS、报警、操作日志和 LLM。"],
      ["loop_master_data", "回路主数据", "维护 PV/SV/MV、PID、阀门、设备和安全约束。"],
      ["policies", "策略与约束", "管理整定上下限、模型先验和安全门禁。"],
      ["users", "角色权限", "配置查看、整定、审批、下发和规则维护权限。"],
    ],
  },
];

const loops = [
  ["5203_TIC_11303", "温度", "分馏塔顶温度", "振荡", 62, "建议整定", "ready", "18.6min 低频振荡"],
  ["5203_LIC_20501A", "液位", "回流罐液位", "迟钝", 58, "暂不整定", "blocked", "疑似阀门死区"],
  ["5203_FIC_20201", "流量", "塔顶回流流量", "健康", 86, "观察", "ready", "运行稳定"],
  ["5203_PIC_20402", "压力", "塔顶压力", "外扰", 69, "先查扰动", "need_more_data", "与压缩机波动相关"],
];

const trendPoints = {
  pv: "M0 110 C60 72 110 70 158 106 S250 145 310 105 S400 68 462 101 S548 141 615 105 S720 72 780 102 S858 140 920 112",
  mv: "M0 166 C70 152 95 137 154 165 S248 198 312 166 S405 134 468 166 S558 196 625 165 S725 135 785 166 S860 196 920 170",
  sp: "M0 101 H920",
};

let activeMenuKey = "workbench";
let activePageKey = "overview";

const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[m]);

function currentMenu() {
  return menus.find((m) => m.key === activeMenuKey) || menus[0];
}

function currentPageTuple() {
  return currentMenu().pages.find(([key]) => key === activePageKey) || currentMenu().pages[0];
}

function render() {
  renderTopMenu();
  renderSideMenu();
  renderHeader();
  $("contentRoot").innerHTML = renderPage(activePageKey);
  bindActions();
}

function renderTopMenu() {
  $("topMenu").innerHTML = menus
    .map((m) => `<button class="top-menu__item ${m.key === activeMenuKey ? "top-menu__item--active" : ""}" data-menu="${m.key}">${m.label}</button>`)
    .join("");
}

function renderSideMenu() {
  const menu = currentMenu();
  $("sideTitle").textContent = menu.label;
  $("sideMenu").innerHTML = menu.pages
    .map(([key, title]) => `<button class="side-item ${key === activePageKey ? "side-item--active" : ""}" data-page="${key}">${title}</button>`)
    .join("");
}

function renderHeader() {
  const menu = currentMenu();
  const [, title, subtitle] = currentPageTuple();
  $("menuEyebrow").textContent = menu.eyebrow;
  $("pageTitle").textContent = title;
  $("pageSubtitle").textContent = subtitle;
  $("pageActions").innerHTML = `
    <button class="secondary-button" data-modal="pageDoc">查看页面说明</button>
    <button class="primary-button" data-modal="runAgent">运行监控 Agent</button>
  `;
}

function kpiCards(items) {
  return `<section class="kpi-grid">${items
    .map(([label, value, desc, tone]) => `<article class="kpi-card ${tone ? `kpi-card--${tone}` : ""}"><span>${label}</span><strong>${value}</strong><p>${desc}</p></article>`)
    .join("")}</section>`;
}

function panel(title, body, extra = "") {
  return `<article class="panel ${extra}"><div class="section-head"><div><p class="eyebrow">Module</p><h3>${title}</h3></div></div>${body}</article>`;
}

function table(headers, rows) {
  return `<div class="table-wrap"><table class="spec-table"><thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows
    .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`)
    .join("")}</tbody></table></div>`;
}

function tag(text, tone = "") {
  return `<span class="badge ${tone ? `badge--${tone}` : ""}">${text}</span>`;
}

function trendChart(title = "PV / SP / MV 趋势", note = "采样 5s · 最近 24h · 模拟数据") {
  return `
    <div class="chart-card">
      <div class="chart-head"><strong>${title}</strong><span>${note}</span></div>
      <svg class="trend-svg" viewBox="0 0 920 230" role="img" aria-label="${title}">
        <path class="grid-line" d="M0 38H920M0 92H920M0 146H920M0 200H920" />
        <path class="sp-line" d="${trendPoints.sp}" />
        <path class="pv-fill" d="${trendPoints.pv} V230 H0Z" />
        <path class="pv-line" d="${trendPoints.pv}" />
        <path class="mv-line" d="${trendPoints.mv}" />
      </svg>
      <div class="legend"><span><i class="legend-pv"></i>PV</span><span><i class="legend-sp"></i>SP</span><span><i class="legend-mv"></i>MV</span></div>
    </div>`;
}

function spectrumChart() {
  return `
    <div class="spectrum">
      ${[20, 36, 74, 128, 48, 31, 22, 18, 12, 9].map((h, i) => `<span style="height:${h}px"><b>${i === 3 ? "18.6m" : ""}</b></span>`).join("")}
    </div>`;
}

function gateList(items) {
  return `<div class="gate-list">${items
    .map(([name, desc, tone]) => `<div class="gate-item gate-item--${tone}"><span></span><div><strong>${name}</strong><p>${desc}</p></div></div>`)
    .join("")}</div>`;
}

function renderLoopTable() {
  return table(["回路", "类型", "对象", "状态", "健康度", "Agent 建议", "准备度", "证据摘要"], loops.map((r) => [
    `<b>${r[0]}</b>`,
    r[1],
    r[2],
    tag(r[3], r[3] === "健康" ? "ok" : r[3] === "振荡" ? "warn" : ""),
    `<b>${r[4]}</b>`,
    r[5],
    r[6],
    r[7],
  ]));
}

function renderPage(key) {
  if (key === "overview") {
    return `
      ${kpiCards([["需要关注", "18", "存在振荡、饱和、迟钝或噪声风险", "danger"], ["建议整定", "7", "参数问题概率高且数据可辨识", "warn"], ["暂不建议整定", "5", "先处理外扰/阀门/仪表", ""], ["健康运行", "46", "运行稳定，无明显退化", "ok"]])}
      <section class="main-grid">${panel("异常回路排行", renderLoopTable(), "panel--large")}${panel("Agent 今日建议", recommendationList())}</section>
      <section class="bottom-grid">${panel("装置级趋势摘要", trendChart("异常回路健康度趋势", "最近 24h · 模拟趋势"))}${panel("本班任务流", taskTimeline())}</section>`;
  }
  if (key === "todos") {
    return `${kpiCards([["待处理", "12", "未关闭的回路待办", "warn"], ["高优先级", "4", "影响生产或质量", "danger"], ["已分派", "6", "已有责任人", "ok"], ["超时", "2", "超过 8h 未处理", "danger"]])}
      ${panel("待处理回路队列", table(["优先级", "回路", "问题", "证据", "建议动作", "责任人", "状态"], [
        ["P1", "5203_TIC_11303", "低频振荡", "PV/MV 同周期，SP 稳定", "进入辨识并生成保守参数", "仪控 A", tag("处理中", "warn")],
        ["P1", "5203_LIC_20501A", "迟钝/死区", "MV 有变化但 PV 响应弱", "检查阀门死区", "设备 B", tag("待确认")],
        ["P2", "5203_PIC_20402", "外扰主导", "与压缩机波动相关", "排查扰动源", "工艺 C", tag("待处理")],
      ]))}`;
  }
  if (key === "shift_tasks") {
    return `${kpiCards([["运行中", "3", "Agent 正在分析", ""], ["待审批", "2", "整定参数等待确认", "warn"], ["失败", "1", "数据源超时", "danger"], ["完成", "9", "本班已关闭", "ok"]])}
      ${panel("任务列表", table(["任务ID", "类型", "回路", "阶段", "开始时间", "耗时", "结果"], [
        ["T-240424-001", "监控分析", "5203_TIC_11303", tag("诊断中", "warn"), "09:12", "42s", "建议整定"],
        ["T-240424-002", "参数候选", "5203_FIC_20201", tag("完成", "ok"), "09:30", "18s", "无需调整"],
        ["T-240424-003", "下发审批", "5203_TIC_11303", tag("待审批", "warn"), "10:05", "3m12s", "等待人工确认"],
      ]))}`;
  }
  if (key === "risk_alerts") {
    return `${kpiCards([["严重", "3", "需本班处理", "danger"], ["警告", "15", "需观察或诊断", "warn"], ["已确认", "8", "已有处理人", "ok"], ["重复风险", "4", "近 7 天多次出现", "danger"]])}
      <section class="main-grid">${panel("风险预警列表", table(["风险", "回路", "严重度", "证据", "建议"], [
        ["持续振荡", "5203_TIC_11303", tag("高", "danger"), "主周期 18.6min，PV/MV 同步", "进入整定准备度评估"],
        ["MV 饱和", "5203_LIC_20501A", tag("高", "danger"), "MV > 95% 持续 32min", "检查阀门裕量"],
        ["数据冻结", "5203_FIC_20201", tag("中", "warn"), "PV 12min 无变化", "检查仪表数据源"],
      ]), "panel--large")}${panel("风险分类", donutLike([["振荡", 42], ["阀门", 28], ["外扰", 18], ["数据", 12]]))}</section>`;
  }

  if (key === "global_board") {
    return `${kpiCards([["监控回路", "74", "已配置完整 PV/SV/MV", "ok"], ["异常", "18", "健康度低于 70", "warn"], ["饱和", "4", "MV 长时间贴边", "danger"], ["自动模式", "93%", "平均自动投用率", "ok"]])}
      ${panel("全局回路看板", renderLoopTable())}`;
  }
  if (key === "loop_profile") {
    return `<section class="main-grid">${panel("单回路画像：5203_TIC_11303", loopProfile(), "panel--large")}${panel("当前 PID 与状态", pidStatus())}</section>
      <section class="bottom-grid">${panel("回路趋势", trendChart())}${panel("上下文与关联", relationCards())}</section>`;
  }
  if (key === "trend_spectrum") {
    return `<section class="main-grid">${panel("趋势分析", trendChart("PV/SV/MV 与负荷趋势", "最近 24h · 自动模式 96.4%"), "panel--large")}${panel("频谱主峰", spectrumChart())}</section>
      ${panel("滞后与相关性", table(["指标", "值", "解释"], [["主周期", "18.6 min", "低频持续振荡"], ["MV→PV 最佳滞后", "65 s", "符合温度回路响应"], ["相关性", "0.71", "MV 动作与 PV 波动强相关"], ["方向", "正作用", "MV 增大后 PV 上升"]]))}`;
  }
  if (key === "alarm_events") {
    return `${panel("事件时间线", table(["时间", "事件", "位号", "说明", "影响判断"], [
      ["08:14", "切自动", "TIC_11303", "自动模式恢复", "可用于后续监控"],
      ["09:47", "高偏差报警", "TIC_11303.PV", "PV 偏离 SP 4.8%", "与振荡峰值一致"],
      ["10:22", "负荷微调", "FIC_20201", "回流量上调 2%", "非主因"],
      ["11:05", "输出接近上限", "LIC_20501A.MV", "MV 达 94%", "疑似阀门裕量不足"],
    ]))}`;
  }

  if (key === "performance_score") {
    return `${kpiCards([["综合性能", "64", "存在明显改善空间", "warn"], ["跟踪评分", "72", "稳态偏差可接受", ""], ["振荡评分", "41", "持续低频振荡", "danger"], ["执行器负担", "58", "MV 行程偏高", "warn"]])}
      <section class="main-grid">${panel("性能指标", table(["指标", "值", "评分", "说明"], [["IAE", "184.2", "62", "偏高"], ["PV 方差", "2.16", "58", "波动偏大"], ["MV travel", "28.4%", "55", "阀门动作偏频繁"], ["Harris 指数", "0.47", "61", "仍有优化空间"]]), "panel--large")}${panel("趋势证据", trendChart())}</section>`;
  }
  if (key === "data_quality") {
    return `${kpiCards([["质量评分", "87", "数据可用于评估和辨识", "ok"], ["缺失率", "0.3%", "低缺失", "ok"], ["异常点", "0.8%", "可接受", ""], ["冻结风险", "低", "未见长时间冻结", "ok"]])}
      ${panel("数据质量明细", table(["检查项", "结果", "阈值", "结论"], [["有效点数", "17280", "> 100", tag("通过", "ok")], ["采样周期", "5.0s ± 0.2s", "抖动 < 10%", tag("通过", "ok")], ["PV span", "5.2℃", "> 噪声 5 倍", tag("通过", "ok")], ["MV span", "8.4%", "> 1%", tag("通过", "ok")], ["噪声/span", "0.11", "< 0.3", tag("通过", "ok")]]))}`;
  }
  if (key === "operating_condition") {
    return `<section class="main-grid">${panel("工况识别结论", conditionPanel(), "panel--large")}${panel("负荷与事件", trendChart("负荷 / SP / PV 趋势", "工况识别窗口"))}</section>
      ${panel("工况证据", table(["证据", "值", "判断"], [["SP 变化次数", "1", "基本稳定"], ["负荷变化率", "0.4%/h", "稳态"], ["报警密度", "低", "非扰动期"], ["操作事件", "无关键改参", "可评价 PID"]]))}`;
  }
  if (key === "readiness") {
    return `${kpiCards([["准备度", "71", "可进入辨识，但需人工确认", "warn"], ["数据质量", "87", "通过", "ok"], ["可辨识性", "76", "通过", "ok"], ["安全门禁", "人工", "禁止自动下发", "warn"]])}
      ${panel("整定准备度门禁", gateList([["数据质量", "有效点数充足，缺失率低", "pass"], ["当前工况", "稳态运行，适合评估", "pass"], ["历史激励", "存在可用窗口，但建议补充最近 7 天", "warn"], ["安全约束", "参数下发必须人工审批", "hold"]]))}`;
  }

  if (key === "diagnosis_overview") {
    return `<section class="main-grid">${panel("根因候选", diagnosisRows(), "panel--large")}${panel("证据趋势", trendChart())}</section>
      ${panel("建议动作", recommendationList())}`;
  }
  if (key === "valve_diagnosis") {
    return `${kpiCards([["饱和比例", "2.1%", "当前不严重", "ok"], ["反向次数", "168", "偏高", "warn"], ["死区评分", "0.31", "证据不足", ""], ["阀位反馈", "未接入", "需补充数据", "warn"]])}
      ${panel("阀门诊断证据", table(["指标", "值", "结论"], [["MV travel", "28.4%", "动作偏多"], ["MV 饱和", "2.1%", "不是主因"], ["stick-slip", "弱", "暂不支持卡涩"], ["阀位反馈", "缺失", "建议接入阀位"]]))}`;
  }
  if (key === "oscillation_diagnosis") {
    return `<section class="main-grid">${panel("振荡诊断", table(["指标", "值", "解释"], [["主周期", "18.6 min", "稳定低频振荡"], ["衰减比", "0.82", "衰减不足"], ["PV/MV 相位", "MV 领先 PV", "更像控制器驱动"], ["同频回路", "无明显", "耦合可能性低"]]), "panel--large")}${panel("频谱", spectrumChart())}</section>`;
  }
  if (key === "model_reliability") {
    return `${kpiCards([["模型置信度", "81%", "可作为整定参考", "ok"], ["R²", "0.855", "拟合较好", "ok"], ["T", "194.7s", "温度回路合理", "ok"], ["风险", "中", "需保守参数", "warn"]])}
      ${panel("模型可靠性诊断", table(["检查项", "值", "结论"], [["模型", "FOPDT", "适合当前数据"], ["K", "0.9595", "方向合理"], ["T", "194.69s", "未命中下界"], ["L", "60.00s", "死区合理"], ["NRMSE", "12.15%", "可接受"], ["竞争模型差距", "fit_score +2.4", "结构较稳定"]]))}`;
  }

  if (key === "tuning_tasks") {
    return `${panel("整定任务列表", table(["任务ID", "来源", "回路", "阶段", "状态", "操作"], [["TT-001", "Agent 建议", "5203_TIC_11303", "参数候选", tag("待确认", "warn"), actionBtn("查看任务", "taskDetail")], ["TT-002", "人工发起", "5203_FIC_20201", "评估", tag("完成", "ok"), actionBtn("查看报告", "taskDetail")], ["TT-003", "定时巡检", "5203_LIC_20501A", "诊断", tag("阻断", "danger"), actionBtn("阻断原因", "blockReason")]]))}`;
  }
  if (key === "identification") {
    return `<section class="main-grid">${panel("候选窗口与趋势", trendChart("辨识窗口 mv_change_4", "窗口 score=0.832 · usable=true"), "panel--large")}${panel("模型结论", modelSummary())}</section>
      ${panel("模型 attempts", table(["Round", "模型", "窗口", "K", "T", "L", "R²", "fit_score", "状态"], [["R0", "FO", "mv_change_4", "0.630", "30.0", "0.0", "0.745", "15.12", tag("选中", "warn")], ["R0", "FOPDT", "mv_change_4", "0.630", "30.0", "0.0", "0.745", "15.11", "成功"], ["R1", "FOPDT", "mv_change_2", "0.959", "194.7", "60.0", "0.855", "18.44", tag("推荐", "ok")]]))}`;
  }
  if (key === "pid_candidates") {
    return `${kpiCards([["推荐策略", "LAMBDA", "保守且稳定", "ok"], ["综合评分", "7.8", "可提交审批", "ok"], ["最大风险", "中", "温度回路响应慢", "warn"], ["PB", "125.5%", "DCS 可用", "ok"]])}
      ${panel("PID 参数候选", table(["策略", "Kp", "PB(%)", "Ki", "Kd", "Ti(s)", "Td(s)", "评价"], [["IMC", "0.9405", "106.33", "0.004831", "28.2147", "194.69", "30.00", "保守"], ["LAMBDA", "0.7967", "125.52", "0.003546", "0.0000", "224.69", "0.00", tag("推荐", "ok")], ["ZN", "4.0583", "24.64", "0.033819", "121.7479", "120.00", "30.00", "偏激进"], ["CHR", "2.0291", "49.28", "0.010422", "60.8739", "194.69", "30.00", "较快"]]))}`;
  }
  if (key === "approval") {
    return `${panel("下发确认单", table(["字段", "当前值", "建议值", "变化", "风险"], [["Kp", "1.20", "0.7967", "-33.6%", "降低振荡风险"], ["PB", "83.33%", "125.52%", "+42.19%", "控制更保守"], ["Ti", "120.0s", "224.69s", "+87.2%", "积分减弱"], ["Td", "0.0s", "0.0s", "不变", "无"]]))}
      ${panel("审批门禁", gateList([["模型可靠性", "置信度 81%，可作为参考", "pass"], ["参数物理约束", "PB/Ti 均在规则范围内", "pass"], ["生产安全", "自动下发关闭，需人工审批", "hold"], ["回滚方案", "保留当前参数，可一键回退", "pass"]]))}`;
  }

  if (key === "cases") {
    return `${panel("整定案例库", table(["案例", "回路", "工况", "诊断", "前后评分", "效果"], [["CASE-118", "5203_TIC_11303", "稳态", "参数偏激进", "5.8 → 7.8", tag("有效", "ok")], ["CASE-109", "5203_FIC_20201", "稳态", "无需整定", "8.6 → 8.7", "观察"], ["CASE-088", "5203_LIC_20501A", "负荷切换", "阀门死区", "4.2 → 6.1", "检修后改善"]]))}`;
  }
  if (key === "rules") {
    return `${panel("规则库", table(["规则", "适用", "当前值", "版本", "状态"], [["temperature_min_T", "温度回路", "30s", "v3", tag("启用", "ok")], ["temperature_Ti_min", "温度整定", "60s", "v3", tag("启用", "ok")], ["PB_range", "全部回路", "5%~1000%", "v2", tag("启用", "ok")], ["auto_writeback", "全部回路", "false", "v1", tag("强制", "warn")]]))}`;
  }
  if (key === "knowledge_graph") {
    return `<section class="main-grid">${panel("知识图谱关系", graphSvg(), "panel--large")}${panel("关联实体", table(["实体", "类型", "关系"], [["5203_TIC_11303", "回路", "控制分馏塔顶温度"], ["V-0202", "设备", "上游影响 LIC20501A"], ["FIC_20201", "回路", "影响回流量"], ["塔顶压力", "变量", "扰动源候选"]]))}</section>`;
  }
  if (key === "model_versions") {
    return `${panel("模型与策略版本", table(["版本", "类型", "范围", "状态", "效果"], [["id-fit-v3", "辨识算法", "全部", tag("生效", "ok"), "T 下界约束已修正"], ["score-v4", "评分规则", "温度/液位", tag("灰度", "warn"), "减少误封顶"], ["agent-prompt-v2", "Agent 策略", "诊断", tag("生效", "ok"), "输出证据链更完整"]]))}`;
  }

  if (key === "data_sources") {
    return `${panel("数据源连接", table(["数据源", "类型", "状态", "延迟", "用途"], [["HistoryDB", "时序库", tag("在线", "ok"), "5s", "PV/SV/MV 趋势"], ["DCS Config", "参数接口", tag("在线", "ok"), "1s", "PID 当前值"], ["AlarmDB", "事件库", tag("在线", "ok"), "10s", "报警/联锁"], ["LLM Gateway", "模型服务", tag("警告", "warn"), "120s timeout", "评审/解释"]]))}`;
  }
  if (key === "loop_master_data") {
    return `${panel("回路主数据", table(["回路", "PV", "SV", "MV", "阀位", "类型", "正反作用"], [["5203_TIC_11303", "TIC11303.PV", "TIC11303.SV", "TIC11303.MV", "TV11303.FB", "temperature", "正作用"], ["5203_LIC_20501A", "LIC20501A.PV", "LIC20501A.SV", "LIC20501A.MV", "LV20501A.FB", "level", "反作用"]]))}`;
  }
  if (key === "policies") {
    return `${panel("策略与约束", table(["策略", "温度", "流量", "压力", "液位"], [["T 最小合理值", "30s", "1s", "5s", "60s"], ["Ti 下限", "60s", "2s", "10s", "60s"], ["PB 范围", "5~1000%", "5~1000%", "5~1000%", "5~1000%"], ["自动下发", "禁止", "禁止", "禁止", "禁止"]]))}`;
  }
  if (key === "users") {
    return `${panel("角色权限", table(["角色", "查看", "发起整定", "审批下发", "维护规则", "维护数据源"], [["运行工程师", "是", "是", "否", "否", "否"], ["仪控工程师", "是", "是", "是", "否", "否"], ["专家管理员", "是", "是", "是", "是", "是"]]))}`;
  }
  return panel("页面待设计", "该页面正在规划中。");
}

function recommendationList() {
  return `<div class="recommendation-list">
    <div class="recommendation recommendation--main"><span>首选</span><h4>进入历史窗口辨识，生成保守 Lambda 参数</h4><p>当前数据质量好，工况稳定，振荡更像参数偏激进。</p></div>
    <div class="recommendation"><span>备选</span><h4>安排 2%~3% 小幅阶跃测试</h4><p>如果最近 7 天窗口激励不足，再安排受控测试。</p></div>
    <div class="recommendation"><span>观察</span><h4>继续采集阀位反馈</h4><p>阀门卡涩证据不足，建议接入阀位后复核。</p></div>
  </div>`;
}

function taskTimeline() {
  return `<div class="agent-steps"><div class="agent-step agent-step--done">08:30 全局巡检完成：18 个异常回路</div><div class="agent-step agent-step--done">09:12 TIC_11303 诊断完成：建议整定</div><div class="agent-step agent-step--active">10:05 参数候选等待审批</div><div class="agent-step">11:00 计划回看效果</div></div>`;
}

function loopProfile() {
  return `${trendChart()}${table(["字段", "值"], [["回路类型", "temperature"], ["控制对象", "分馏塔顶温度"], ["自动模式比例", "96.4%"], ["当前状态", "低频振荡"], ["Agent 结论", "建议进入整定准备流程"]])}`;
}

function pidStatus() {
  return table(["参数", "当前值"], [["Kp", "1.2000"], ["PB", "83.33%"], ["Ki", "0.010000"], ["Kd", "0.0000"], ["Ti", "120.0s"], ["Td", "0.0s"], ["输出限幅", "0%~100%"], ["正反作用", "正作用"]]);
}

function relationCards() {
  return `<div class="data-source-list"><div><b>上游变量</b><span>回流流量 FIC_20201、塔顶压力 PIC_20402</span></div><div><b>下游影响</b><span>产品质量软测量、塔顶冷凝负荷</span></div><div><b>关联设备</b><span>分馏塔、冷凝器、回流罐</span></div><div><b>同类案例</b><span>CASE-118、CASE-094</span></div></div>`;
}

function conditionPanel() {
  return `<div class="decision-card"><h4>当前工况：稳态运行</h4><p>SP 基本不变，负荷变化率 0.4%/h，未处于开停车或牌号切换。可以评价当前 PID 控制效果。</p>${gateList([["SP 稳定", "24h 内仅 1 次微调", "pass"], ["负荷稳定", "无明显 ramp", "pass"], ["扰动较弱", "报警密度低", "pass"]])}</div>`;
}

function diagnosisRows() {
  return `<div class="diagnosis-list">
    <div class="diagnosis-row diagnosis-row--primary"><div><strong>PID 参数偏激进</strong><p>PV/MV 同周期振荡，MV 未饱和，扰动事件少。</p></div><span>0.78</span></div>
    <div class="diagnosis-row"><div><strong>外部周期扰动</strong><p>有少量负荷波动，但与主周期不完全一致。</p></div><span>0.42</span></div>
    <div class="diagnosis-row"><div><strong>阀门卡涩/死区</strong><p>缺少阀位反馈，当前证据不足。</p></div><span>0.31</span></div>
  </div>`;
}

function modelSummary() {
  return table(["字段", "值"], [["推荐模型", "FOPDT"], ["K", "0.9595"], ["T", "194.69s"], ["L", "60.00s"], ["R²", "0.855"], ["NRMSE", "12.15%"], ["置信度", "81%"]]);
}

function donutLike(items) {
  return `<div class="donut-list">${items.map(([name, value]) => `<div><span>${name}</span><b style="width:${value}%"></b><em>${value}%</em></div>`).join("")}</div>`;
}

function graphSvg() {
  return `<svg class="graph-svg" viewBox="0 0 560 260"><line x1="110" y1="130" x2="280" y2="70"/><line x1="110" y1="130" x2="280" y2="190"/><line x1="280" y1="70" x2="450" y2="130"/><line x1="280" y1="190" x2="450" y2="130"/><circle cx="110" cy="130" r="54"/><circle cx="280" cy="70" r="48"/><circle cx="280" cy="190" r="48"/><circle cx="450" cy="130" r="54"/><text x="110" y="126">TIC</text><text x="110" y="146">11303</text><text x="280" y="74">FIC</text><text x="280" y="194">PIC</text><text x="450" y="126">塔顶</text><text x="450" y="146">质量</text></svg>`;
}

function actionBtn(text, modal) {
  return `<button class="inline-button" data-modal="${modal}">${text}</button>`;
}

function bindActions() {
  document.querySelectorAll("[data-menu]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeMenuKey = btn.dataset.menu;
      activePageKey = currentMenu().pages[0][0];
      render();
    });
  });
  document.querySelectorAll("[data-page]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activePageKey = btn.dataset.page;
      render();
    });
  });
  document.querySelectorAll("[data-modal]").forEach((btn) => {
    btn.addEventListener("click", () => openModal(btn.dataset.modal));
  });
}

function openModal(kind) {
  const page = currentPageTuple();
  const content = {
    pageDoc: `<p>当前页面：<b>${page[1]}</b></p><p>${page[2]}</p><p>正式开发时建议拆为筛选区、KPI 区、主图表区、明细表格区、证据/建议区。</p>`,
    runAgent: `<p>模拟运行监控 Agent：</p>${gateList([["拉取历史数据", "最近 24h PV/SV/MV、报警、PID 参数", "pass"], ["评估工况", "识别为稳态运行", "pass"], ["诊断状态", "检测低频振荡", "warn"], ["输出建议", "建议进入整定准备流程", "hold"]])}`,
    taskDetail: `<p>任务详情弹窗用于承载完整事件流、LLM 思维链、输入输出 JSON、失败原因和重试按钮，避免主页面过长。</p>`,
    blockReason: `<p>阻断原因：当前疑似阀门死区，历史数据不足以证明 PID 参数是主因。建议先接入阀位反馈或安排现场检查。</p>`,
  }[kind] || `<p>功能弹窗示例。</p>`;
  $("modalTitle").textContent = kind === "runAgent" ? "运行监控 Agent" : "页面/任务详情";
  $("modalBody").innerHTML = content;
  $("modalBackdrop").hidden = false;
}

$("modalClose").addEventListener("click", () => ($("modalBackdrop").hidden = true));
$("modalBackdrop").addEventListener("click", (event) => {
  if (event.target === $("modalBackdrop")) $("modalBackdrop").hidden = true;
});

render();
