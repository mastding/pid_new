// 生成 PPT：
//   Slide 1：PID 整定的技术发展路线（5 个里程碑时间线）
//   Slide 2：当前 PID 整定系统的痛点（2x3 卡片矩阵）
//   Slide 3：我们的解决方案 · 本体驱动 + LLM 智能整定（2x3 卡片，与痛点一一对应）
//
// 配色：Midnight Executive 主色 + Coral（痛点）+ Teal（解法）；中文字体微软雅黑。
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const {
  FaWrench,
  FaCogs,
  FaChartLine,
  FaBrain,
  FaProjectDiagram,
  FaIndustry,
  FaSnowflake,
  FaQuestionCircle,
  FaUserMd,
  FaUnlink,
  FaUserTie,
  FaSitemap,
  FaLayerGroup,
  FaSearchPlus,
  FaStethoscope,
  FaSyncAlt,
  FaArchive,
} = require("react-icons/fa");

// ── 调色板 ──────────────────────────────────────────────────────────────
const COL = {
  navy: "1E2761",       // 主背景
  navyDeep: "0F1A47",   // 更深的导航蓝
  ice: "CADCFC",        // 浅蓝（副色）
  coral: "F96167",      // 强调红（痛点）
  amber: "F2B53C",      // 强调金（里程碑）
  teal: "0D9488",       // 强调绿松石（解法）
  tealLight: "E0F7F4",  // 解法卡片底
  tealBorder: "B7E5DD", // 解法卡片边框
  white: "FFFFFF",
  textDim: "8A99B8",    // 暗背景上的次要文字
  bodyDark: "1F2937",   // 浅背景上的正文
  bodyMute: "475569",   // 浅背景上的次要文字
  cardBg: "F6F8FE",     // 浅卡片底
  cardBorder: "DCE3F2", // 浅卡片边框
};

const FONT_HEAD = "微软雅黑";
const FONT_BODY = "微软雅黑";

async function iconPng(IconComp, color, size = 320) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComp, { color, size: String(size) })
  );
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
  pres.title = "PID 智能整定 · 技术演进与现状痛点";
  pres.author = "PID V2";

  // 预先把所有图标渲染好
  const ic = {
    wrench: await iconPng(FaWrench, "#" + COL.coral),
    cogs: await iconPng(FaCogs, "#" + COL.coral),
    chart: await iconPng(FaChartLine, "#" + COL.coral),
    brain: await iconPng(FaBrain, "#" + COL.coral),
    graph: await iconPng(FaBrain, "#" + COL.amber), // 最后一个里程碑用金色
    industry: await iconPng(FaIndustry, "#" + COL.coral),
    snow: await iconPng(FaSnowflake, "#" + COL.coral),
    question: await iconPng(FaQuestionCircle, "#" + COL.coral),
    userMd: await iconPng(FaUserMd, "#" + COL.coral),
    unlink: await iconPng(FaUnlink, "#" + COL.coral),
    userTie: await iconPng(FaUserTie, "#" + COL.coral),
    projectDiagram: await iconPng(FaProjectDiagram, "#" + COL.amber),
    // 解法页：白色图标（衬深色圆底）
    solSitemap: await iconPng(FaSitemap, "#" + COL.white),
    solLayer: await iconPng(FaLayerGroup, "#" + COL.white),
    solSearch: await iconPng(FaSearchPlus, "#" + COL.white),
    solSteth: await iconPng(FaStethoscope, "#" + COL.white),
    solSync: await iconPng(FaSyncAlt, "#" + COL.white),
    solArchive: await iconPng(FaArchive, "#" + COL.white),
  };

  // ─────────────────────────────────────────────────────────────────────
  // SLIDE 1：PID 整定的技术发展路线
  // ─────────────────────────────────────────────────────────────────────
  const s1 = pres.addSlide();
  s1.background = { color: COL.navy };

  // 顶部装饰：薄薄一条 ice 色细线，左对齐 0.6"，让标题区有"工业仪表"感
  s1.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 0.55, w: 0.5, h: 0.1,
    fill: { color: COL.coral }, line: { color: COL.coral },
  });

  // 标题
  s1.addText("PID 整定技术发展路线", {
    x: 0.6, y: 0.7, w: 12.1, h: 0.7,
    fontSize: 34, bold: true, color: COL.white, fontFace: FONT_HEAD,
    margin: 0,
  });
  // 副标题
  s1.addText("从经验法则到本体驱动 · 八十年的技术演进", {
    x: 0.6, y: 1.4, w: 12.1, h: 0.4,
    fontSize: 14, color: COL.ice, fontFace: FONT_BODY, italic: true, margin: 0,
  });

  // 时间线：5 个里程碑横向排开
  const milestones = [
    {
      year: "1942",
      era: "经验法",
      title: "Ziegler-Nichols",
      desc: "闭环临界比例度法、开环阶跃响应法。完全靠工程师手算 + 现场试凑。",
      icon: ic.wrench,
    },
    {
      year: "1980s",
      era: "模型法",
      title: "IMC / Lambda / CHR",
      desc: "FOPDT/SOPDT 过程模型 + 解析公式。计算机辅助整定软件出现。",
      icon: ic.cogs,
    },
    {
      year: "1990s+",
      era: "自整定 + 频域",
      title: "Relay Feedback / 自适应",
      desc: "在线辨识 + 频域分析。商业软件（ExperTune 等）让现场工程师可一键整定。",
      icon: ic.chart,
    },
    {
      year: "2010s+",
      era: "数据驱动 + AI",
      title: "黑箱建模 + 性能基准",
      desc: "Subspace / ARMAX 系统辨识；Harris 指数、Cpk 评估闭环健康度。",
      icon: ic.brain,
    },
    {
      year: "现在",
      era: "本体 + LLM",
      title: "知识增强智能整定",
      desc: "工艺本体（MCP）+ LLM 顾问选窗 / 评审 / 精修。可解释、可重放、可降级。",
      icon: ic.projectDiagram,
      highlight: true,
    },
  ];

  // 时间线参数
  const tlY = 4.4;            // 主线 y
  const tlLeft = 0.7;
  const tlRight = 12.6;
  const cardW = 2.36;
  const cardGap = ((tlRight - tlLeft) - cardW * milestones.length) / (milestones.length - 1);
  const cardTop = 2.2;
  const cardH = 1.95;

  // 主时间线（细线）
  s1.addShape(pres.shapes.LINE, {
    x: tlLeft, y: tlY, w: tlRight - tlLeft, h: 0,
    line: { color: COL.ice, width: 1.25 },
  });

  for (let i = 0; i < milestones.length; i++) {
    const m = milestones[i];
    const cardX = tlLeft + i * (cardW + cardGap);
    const dotX = cardX + cardW / 2 - 0.13; // dot center to card center
    const dotColor = m.highlight ? COL.amber : COL.coral;

    // 时间点圆点
    s1.addShape(pres.shapes.OVAL, {
      x: dotX, y: tlY - 0.13, w: 0.26, h: 0.26,
      fill: { color: dotColor }, line: { color: dotColor },
    });
    // 圆点外圈轻晕
    s1.addShape(pres.shapes.OVAL, {
      x: dotX - 0.1, y: tlY - 0.23, w: 0.46, h: 0.46,
      fill: { color: dotColor, transparency: 75 }, line: { color: dotColor, transparency: 75 },
    });

    // 圆点 → 卡片连接小杆
    s1.addShape(pres.shapes.LINE, {
      x: dotX + 0.13, y: tlY - 0.13, w: 0, h: -(tlY - 0.13 - (cardTop + cardH)),
      line: { color: COL.ice, width: 0.75, transparency: 30 },
    });

    // 年份（在主线下方）
    s1.addText(m.year, {
      x: cardX, y: tlY + 0.18, w: cardW, h: 0.32,
      align: "center", fontSize: 13, bold: true,
      color: m.highlight ? COL.amber : COL.coral, fontFace: FONT_HEAD, margin: 0,
    });
    // 时代标签
    s1.addText(m.era, {
      x: cardX, y: tlY + 0.5, w: cardW, h: 0.3,
      align: "center", fontSize: 11, color: COL.ice, fontFace: FONT_BODY, margin: 0,
    });

    // 卡片：浅描边方块
    s1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: cardX, y: cardTop, w: cardW, h: cardH,
      fill: { color: COL.navyDeep },
      line: { color: m.highlight ? COL.amber : COL.ice, width: m.highlight ? 1.5 : 0.5, transparency: m.highlight ? 0 : 60 },
      rectRadius: 0.08,
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.25 },
    });

    // 卡片内图标
    s1.addImage({ data: m.icon, x: cardX + 0.18, y: cardTop + 0.18, w: 0.4, h: 0.4 });
    // 卡片内标题
    s1.addText(m.title, {
      x: cardX + 0.65, y: cardTop + 0.13, w: cardW - 0.75, h: 0.5,
      fontSize: 13.5, bold: true,
      color: m.highlight ? COL.amber : COL.white, fontFace: FONT_HEAD, margin: 0,
    });
    // 描述
    s1.addText(m.desc, {
      x: cardX + 0.18, y: cardTop + 0.7, w: cardW - 0.36, h: cardH - 0.85,
      fontSize: 10.5, color: m.highlight ? COL.white : COL.ice,
      fontFace: FONT_BODY, valign: "top", margin: 0, paraSpaceAfter: 2,
    });
  }

  // 底部一句旁白（"我们处于"）
  s1.addText([
    { text: "我们正处于 ", options: { color: COL.ice } },
    { text: "本体 + LLM ", options: { color: COL.amber, bold: true } },
    { text: "驱动整定的起步期：算法 80 年的成果是地基，知识与决策的结构化才是新跃迁。", options: { color: COL.ice } },
  ], {
    x: 0.7, y: 6.55, w: 11.9, h: 0.45,
    fontSize: 13, fontFace: FONT_BODY, italic: true, margin: 0, valign: "middle",
  });

  // 页脚
  s1.addText("PID 整定技术演进 · 1/2", {
    x: 0.6, y: 7.05, w: 6, h: 0.3,
    fontSize: 9, color: COL.textDim, fontFace: FONT_BODY, margin: 0,
  });

  // ─────────────────────────────────────────────────────────────────────
  // SLIDE 2：当前 PID 整定系统的痛点（2x3 卡片）
  // ─────────────────────────────────────────────────────────────────────
  const s2 = pres.addSlide();
  s2.background = { color: COL.white };

  // 顶部色块标题区
  s2.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 13.3, h: 1.4,
    fill: { color: COL.navy }, line: { color: COL.navy },
  });
  s2.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 0.5, w: 0.5, h: 0.1,
    fill: { color: COL.coral }, line: { color: COL.coral },
  });
  s2.addText("当前 PID 整定系统的核心痛点", {
    x: 0.6, y: 0.65, w: 12.1, h: 0.55,
    fontSize: 30, bold: true, color: COL.white, fontFace: FONT_HEAD, margin: 0,
  });
  s2.addText("算法越做越复杂，工程问题却没有被真正解决", {
    x: 0.6, y: 1.18, w: 12.1, h: 0.32,
    fontSize: 13, color: COL.ice, fontFace: FONT_BODY, italic: true, margin: 0,
  });

  // 6 个痛点卡片，2 行 x 3 列
  const pains = [
    {
      icon: ic.unlink,
      title: "工艺与数据脱节",
      stat: "01",
      desc: "整定算法只看 PV/MV 数值，不知道当前回路的物理意义、量程、USL/LSL、阀门特性、安全联锁。算出「漂亮的数学解」和现场不可用之间常常没有边界。",
    },
    {
      icon: ic.snow,
      title: "参数静态、工况盲",
      stat: "02",
      desc: "一次整定一组参数。负荷切换、季节变化、催化剂老化后控制器逐渐「失效」，但没人重新调整。整定的成果在现场只有「出生那一刻」是匹配的。",
    },
    {
      icon: ic.question,
      title: "选窗与决策黑箱",
      stat: "03",
      desc: "算法挑出「分数最高」的窗口和模型，但工程师无法判断它是否符合工艺直觉。「为什么是这一段？」回答只有一行 R²，缺乏可追溯证据。",
    },
    {
      icon: ic.userMd,
      title: "缺乏诊断前置",
      stat: "04",
      desc: "直接发起整定，不分辨阀门粘滞、控制器开环、测量噪声异常。在不该整定的回路上整定，结果是一组奇怪参数被工程师悄悄忽略。",
    },
    {
      icon: ic.chart,
      title: "整定与运行脱节",
      stat: "05",
      desc: "PID 参数下发到 DCS 后，没有谁持续盯着 Cpk / Harris 指数是否真的提升。「整定有没有用」通常要等故障再次发生才被发现。",
    },
    {
      icon: ic.userTie,
      title: "高度依赖个人经验",
      stat: "06",
      desc: "整定质量取决于工艺老师傅在不在场。换一个工程师就要从头试凑；班组的整定经验既不可复用，也无法在系统里沉淀。",
    },
  ];

  const cardCols = 3;
  const cardRows = 2;
  const padX = 0.6;
  const padY = 1.7;       // 标题区下方留白
  const gridW = 13.3 - padX * 2;
  const gridH = 7.5 - padY - 0.55; // 留出底部一行洞察
  const cardGapX = 0.3;
  const cardGapY = 0.3;
  const cardCellW = (gridW - cardGapX * (cardCols - 1)) / cardCols;
  const cardCellH = (gridH - cardGapY * (cardRows - 1)) / cardRows;

  for (let i = 0; i < pains.length; i++) {
    const r = Math.floor(i / cardCols);
    const c = i % cardCols;
    const x = padX + c * (cardCellW + cardGapX);
    const y = padY + r * (cardCellH + cardGapY);
    const p = pains[i];

    // 卡片底
    s2.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: cardCellW, h: cardCellH,
      fill: { color: COL.cardBg },
      line: { color: COL.cardBorder, width: 0.75 },
      rectRadius: 0.08,
    });

    // 左侧深色色条
    s2.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.12, h: cardCellH,
      fill: { color: COL.coral }, line: { color: COL.coral },
    });

    // 右上角的编号（半透明大字）
    s2.addText(p.stat, {
      x: x + cardCellW - 1.1, y: y + 0.1, w: 1, h: 0.7,
      fontSize: 40, bold: true, italic: true,
      color: COL.cardBorder, fontFace: FONT_HEAD, align: "right", margin: 0,
    });

    // 图标圆底
    s2.addShape(pres.shapes.OVAL, {
      x: x + 0.35, y: y + 0.35, w: 0.6, h: 0.6,
      fill: { color: COL.navy }, line: { color: COL.navy },
    });
    s2.addImage({ data: p.icon, x: x + 0.485, y: y + 0.485, w: 0.33, h: 0.33 });

    // 标题
    s2.addText(p.title, {
      x: x + 1.1, y: y + 0.32, w: cardCellW - 1.3, h: 0.5,
      fontSize: 17, bold: true, color: COL.bodyDark, fontFace: FONT_HEAD, margin: 0,
    });

    // 描述
    s2.addText(p.desc, {
      x: x + 0.35, y: y + 1.1, w: cardCellW - 0.6, h: cardCellH - 1.2,
      fontSize: 11.5, color: COL.bodyMute, fontFace: FONT_BODY,
      valign: "top", margin: 0, paraSpaceAfter: 2,
    });
  }

  // 底部洞察
  s2.addText([
    { text: "本质：", options: { color: COL.coral, bold: true } },
    { text: "整定 ≠ 求解 PID 三个参数。整定是 ", options: { color: COL.bodyDark } },
    { text: "工艺知识 + 数据画像 + 决策证据 ", options: { color: COL.navy, bold: true } },
    { text: "持续闭环的系统工程。", options: { color: COL.bodyDark } },
  ], {
    x: padX, y: 7.1, w: 13.3 - padX * 2, h: 0.32,
    fontSize: 13, fontFace: FONT_BODY, italic: true, align: "left", margin: 0, valign: "middle",
  });

  // ─────────────────────────────────────────────────────────────────────
  // SLIDE 3：解决方案 · 本体驱动 + LLM 智能整定（与痛点 1-6 一一映射）
  // ─────────────────────────────────────────────────────────────────────
  const s3 = pres.addSlide();
  s3.background = { color: COL.white };

  // 顶部色块标题区（深底 + 一抹 teal）
  s3.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 13.3, h: 1.4,
    fill: { color: COL.navy }, line: { color: COL.navy },
  });
  s3.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 0.5, w: 0.5, h: 0.1,
    fill: { color: COL.teal }, line: { color: COL.teal },
  });
  s3.addText("我们的解决方案 · 本体驱动 + LLM 智能整定", {
    x: 0.6, y: 0.65, w: 12.1, h: 0.55,
    fontSize: 28, bold: true, color: COL.white, fontFace: FONT_HEAD, margin: 0,
  });
  s3.addText("把工艺知识、数据画像、决策证据搬进同一条流水线", {
    x: 0.6, y: 1.18, w: 12.1, h: 0.32,
    fontSize: 13, color: COL.ice, fontFace: FONT_BODY, italic: true, margin: 0,
  });

  // 6 张解法卡片，与痛点 1-6 一一映射
  const sols = [
    {
      icon: ic.solSitemap,
      from: "痛点 01",
      title: "本体 / MCP 注入",
      desc: "整定开始即从本体拉取 USL/LSL、量程、过程方向、阀门特性。算法和 LLM 都看到工艺，不再「盲算」。",
    },
    {
      icon: ic.solLayer,
      from: "痛点 02",
      title: "工况识别 + 多工况策略",
      desc: "诊断阶段先识别当前工况（高/低负荷、过渡），按工况切换阈值与时间常数先验，避免单一参数走天下。",
    },
    {
      icon: ic.solSearch,
      from: "痛点 03",
      title: "5 算法族 + LLM 评审",
      desc: "每个候选窗口都有原始分、本体一致性分、策略分；LLM 逐窗给 verdict + 理由 + 本体证据，决策全程可追溯。",
    },
    {
      icon: ic.solSteth,
      from: "痛点 04",
      title: "诊断先于整定",
      desc: "独立诊断流水线先跑开闭环判定、SNR、Cpk、Harris；只有「需要整定」的回路才进入辨识 + 整定主流程。",
    },
    {
      icon: ic.solSync,
      from: "痛点 05",
      title: "整定后持续监控",
      desc: "参数下发后监控页持续跟踪 Cpk / Harris / 过冲；性能下滑会自动回到诊断入口，形成「整定—运行—再整定」闭环。",
    },
    {
      icon: ic.solArchive,
      from: "痛点 06",
      title: "经验沉淀 + 会话可重放",
      desc: "每次整定的本体证据、思维链、攻坚精修都进会话日志；专家知识沉到本体可复用，新工程师也能站在前人结论上。",
    },
  ];

  // 复用与 slide 2 相同的网格布局
  for (let i = 0; i < sols.length; i++) {
    const r = Math.floor(i / cardCols);
    const c = i % cardCols;
    const x = padX + c * (cardCellW + cardGapX);
    const y = padY + r * (cardCellH + cardGapY);
    const sol = sols[i];

    // 卡片底（淡 teal）
    s3.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: cardCellW, h: cardCellH,
      fill: { color: COL.tealLight },
      line: { color: COL.tealBorder, width: 0.75 },
      rectRadius: 0.08,
    });

    // 左侧 teal 色条
    s3.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.12, h: cardCellH,
      fill: { color: COL.teal }, line: { color: COL.teal },
    });

    // 右上角的"痛点 0X →"映射标签
    s3.addText([
      { text: sol.from, options: { color: COL.bodyMute } },
      { text: "  →", options: { color: COL.teal, bold: true } },
    ], {
      x: x + cardCellW - 1.7, y: y + 0.18, w: 1.55, h: 0.32,
      fontSize: 11, fontFace: FONT_BODY, italic: true, align: "right", margin: 0,
    });

    // 图标（teal 深色圆底 + 白图标）
    s3.addShape(pres.shapes.OVAL, {
      x: x + 0.35, y: y + 0.35, w: 0.6, h: 0.6,
      fill: { color: COL.teal }, line: { color: COL.teal },
    });
    s3.addImage({ data: sol.icon, x: x + 0.485, y: y + 0.485, w: 0.33, h: 0.33 });

    // 标题
    s3.addText(sol.title, {
      x: x + 1.1, y: y + 0.32, w: cardCellW - 1.3, h: 0.5,
      fontSize: 17, bold: true, color: COL.bodyDark, fontFace: FONT_HEAD, margin: 0,
    });

    // 描述
    s3.addText(sol.desc, {
      x: x + 0.35, y: y + 1.1, w: cardCellW - 0.6, h: cardCellH - 1.2,
      fontSize: 11.5, color: COL.bodyMute, fontFace: FONT_BODY,
      valign: "top", margin: 0, paraSpaceAfter: 2,
    });
  }

  // 底部一句话定位
  s3.addText([
    { text: "底层逻辑：", options: { color: COL.teal, bold: true } },
    { text: "从「算 PID 三个参数」到「", options: { color: COL.bodyDark } },
    { text: "工艺知识 + 数据画像 + 决策证据持续闭环", options: { color: COL.navy, bold: true } },
    { text: "」的整定系统。", options: { color: COL.bodyDark } },
  ], {
    x: padX, y: 7.1, w: 13.3 - padX * 2, h: 0.32,
    fontSize: 13, fontFace: FONT_BODY, italic: true, align: "left", margin: 0, valign: "middle",
  });

  // ─────────────────────────────────────────────────────────────────────
  await pres.writeFile({ fileName: "../PID整定_技术演进与痛点.pptx" });
  console.log("ok: ../PID整定_技术演进与痛点.pptx");
})();
