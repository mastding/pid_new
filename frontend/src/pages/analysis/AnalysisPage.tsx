import { useState, useCallback, useMemo } from 'react';
import { PageContainer, ProCard } from '@ant-design/pro-components';
import {
  Upload,
  Button,
  Space,
  Select,
  Table,
  Tag,
  message,
  Empty,
  Descriptions,
  Alert,
  DatePicker,
  Slider,
} from 'antd';
import { UploadOutlined, SearchOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd';
import type { Dayjs } from 'dayjs';
import dayjs from 'dayjs';
import { Line } from '@ant-design/charts';
import { getLoopSeries, inspectLoops, inspectWindows, inferLoopTypeFromPrefix } from '@/services/api';
import type { LoopSeriesResp } from '@/services/api';

const LOOP_TYPE_OPTIONS = [
  { label: '流量 (flow)', value: 'flow' },
  { label: '压力 (pressure)', value: 'pressure' },
  { label: '温度 (temperature)', value: 'temperature' },
  { label: '液位 (level)', value: 'level' },
];
import type { CandidateWindow } from '@/types/tuning';

interface LoopInfo {
  prefix: string;
  pv_col: string;
  mv_col: string;
  sv_col: string;
}

interface InspectLoopsResp {
  loops: LoopInfo[];
  total_rows: number;
  sampling_time: number;
  csv_path: string;
  error?: string;
}

interface InspectWindowsResp {
  windows: CandidateWindow[];
  total_rows: number;
  sampling_time: number;
  step_events: number;
  usable_count: number;
  csv_path: string;
  error?: string;
}

export default function AnalysisPage() {
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [loops, setLoops] = useState<LoopInfo[]>([]);
  const [selectedLoop, setSelectedLoop] = useState<string>('');
  const [loopType, setLoopType] = useState<string>('');
  const [loopTypeInferred, setLoopTypeInferred] = useState<boolean>(false);
  const [windows, setWindows] = useState<CandidateWindow[]>([]);
  const [selectedWindow, setSelectedWindow] = useState<CandidateWindow | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [csvPath, setCsvPath] = useState<string>('');
  const [series, setSeries] = useState<LoopSeriesResp | null>(null);
  const [seriesLoading, setSeriesLoading] = useState(false);
  const [seriesError, setSeriesError] = useState<string | null>(null);
  const [timeRangeTs, setTimeRangeTs] = useState<[Dayjs, Dayjs] | null>(null);
  const [timeRangeT, setTimeRangeT] = useState<[number, number] | null>(null);
  const [meta, setMeta] = useState<{
    rows: number;
    dt: number;
    steps?: number;
    usable?: number;
  } | null>(null);

  const loadSeries = useCallback(async (cp: string, prefix: string) => {
    if (!cp) return;
    setSeriesLoading(true);
    setSeriesError(null);
    setSeries(null);
    setTimeRangeTs(null);
    setTimeRangeT(null);
    try {
      const resp: LoopSeriesResp = await getLoopSeries({
        csv_path: cp,
        loop_prefix: prefix || undefined,
        max_points: 4000,
      });
      if (resp.error) {
        setSeriesError(resp.error);
        return;
      }
      setSeries(resp);
      if (resp.points.length > 0) {
        if (resp.x_axis === 'timestamp') {
          const start = dayjs(String(resp.points[0].t));
          const end = dayjs(String(resp.points[resp.points.length - 1].t));
          if (start.isValid() && end.isValid()) setTimeRangeTs([start, end]);
        } else {
          const minT = Number(resp.points[0].t);
          const maxT = Number(resp.points[resp.points.length - 1].t);
          if (Number.isFinite(minT) && Number.isFinite(maxT)) setTimeRangeT([minT, maxT]);
        }
      }
    } catch (e) {
      setSeriesError(String(e));
    } finally {
      setSeriesLoading(false);
    }
  }, []);

  const handleInspect = useCallback(async () => {
    const file = fileList[0]?.originFileObj as File | undefined;
    if (!file) {
      message.warning('请先上传 CSV 文件');
      return;
    }
    setLoading(true);
    setError(null);
    setCsvPath('');
    setSeries(null);
    setSeriesError(null);
    setLoops([]);
    setWindows([]);
    setSelectedWindow(null);

    try {
      const resp: InspectLoopsResp = await inspectLoops(file);
      if (resp.error) {
        setError(resp.error);
        return;
      }
      setLoops(resp.loops);
      setMeta({ rows: resp.total_rows, dt: resp.sampling_time });

      const firstPrefix = resp.loops[0]?.prefix || '';
      setSelectedLoop(firstPrefix);

      // 推断回路类型：能从前缀识别到则自动用，否则保持空让用户在下拉框选
      const inferred = inferLoopTypeFromPrefix(firstPrefix);
      const effectiveLoopType = inferred ?? loopType;
      setLoopType(effectiveLoopType);
      setLoopTypeInferred(inferred !== null);

      // 自动加载第一个回路的窗口
      const wResp: InspectWindowsResp = await inspectWindows(
        file, firstPrefix || undefined, effectiveLoopType || undefined,
      );
      if (wResp.error) {
        setError(wResp.error);
      } else {
        setWindows(wResp.windows);
        setCsvPath(wResp.csv_path);
        setMeta({
          rows: wResp.total_rows,
          dt: wResp.sampling_time,
          steps: wResp.step_events,
          usable: wResp.usable_count,
        });
        await loadSeries(wResp.csv_path, firstPrefix);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [fileList, loadSeries]);

  const reloadWindows = useCallback(
    async (prefix: string, lt: string) => {
      const file = fileList[0]?.originFileObj as File | undefined;
      if (!file) return;
      setLoading(true);
      try {
        const wResp: InspectWindowsResp = await inspectWindows(
          file, prefix || undefined, lt || undefined,
        );
        if (wResp.error) setError(wResp.error);
        else {
          setWindows(wResp.windows);
          setSelectedWindow(null);
          setCsvPath(wResp.csv_path);
          setMeta({
            rows: wResp.total_rows,
            dt: wResp.sampling_time,
            steps: wResp.step_events,
            usable: wResp.usable_count,
          });
          await loadSeries(wResp.csv_path, prefix);
        }
      } finally {
        setLoading(false);
      }
    },
    [fileList, loadSeries],
  );

  const handleSwitchLoop = useCallback(
    async (prefix: string) => {
      setSelectedLoop(prefix);
      const inferred = inferLoopTypeFromPrefix(prefix);
      const effective = inferred ?? loopType;
      setLoopType(effective);
      setLoopTypeInferred(inferred !== null);
      await reloadWindows(prefix, effective);
    },
    [loopType, reloadWindows],
  );

  const handleSwitchLoopType = useCallback(
    async (lt: string) => {
      setLoopType(lt);
      setLoopTypeInferred(false);  // 用户手动改了，标记非自动
      await reloadWindows(selectedLoop, lt);
    },
    [selectedLoop, reloadWindows],
  );

  // 构造预览图数据
  const previewData = selectedWindow
    ? (() => {
        const dt = meta?.dt ?? 1;
        const pv = (selectedWindow as CandidateWindow & { preview_pv?: number[] }).preview_pv ?? [];
        const mv = (selectedWindow as CandidateWindow & { preview_mv?: number[] }).preview_mv ?? [];
        const n = Math.min(pv.length, mv.length);
        const data: { t: number; value: number; series: string }[] = [];
        const stride = (selectedWindow.end - selectedWindow.start) / Math.max(n, 1);
        for (let i = 0; i < n; i++) {
          const t = parseFloat((i * stride * dt).toFixed(1));
          data.push({ t, value: pv[i], series: 'PV' });
          data.push({ t, value: mv[i], series: 'MV' });
        }
        return data;
      })()
    : [];

  const seriesPlotData = useMemo(() => {
    if (!series || !series.points?.length) return [];
    let pts = series.points;
    if (series.x_axis === 'timestamp' && timeRangeTs) {
      const [start, end] = timeRangeTs;
      pts = pts.filter((p) => {
        const t = dayjs(String(p.t));
        return t.isValid() && (t.isAfter(start) || t.isSame(start)) && (t.isBefore(end) || t.isSame(end));
      });
    }
    if (series.x_axis === 't' && timeRangeT) {
      const [start, end] = timeRangeT;
      pts = pts.filter((p) => {
        const t = Number(p.t);
        return Number.isFinite(t) && t >= start && t <= end;
      });
    }

    const data: { t: string | number; value: number; series: string }[] = [];
    for (const p of pts) {
      data.push({ t: p.t, value: p.pv, series: 'PV' });
      if (p.sv !== null && p.sv !== undefined) data.push({ t: p.t, value: p.sv, series: 'SV' });
      data.push({ t: p.t, value: p.mv, series: 'MV' });
    }
    return data;
  }, [series, timeRangeT, timeRangeTs]);

  return (
    <PageContainer title="回路分析" subTitle="上传 CSV → 检测回路 → 浏览候选辨识窗口">
      <ProCard style={{ marginBottom: 16 }}>
        <Space size="large" align="center">
          <Upload
            accept=".csv"
            maxCount={1}
            fileList={fileList}
            beforeUpload={() => false}
            onChange={({ fileList: fl }) => setFileList(fl)}
          >
            <Button icon={<UploadOutlined />}>选择 CSV 文件</Button>
          </Upload>
          <Button
            type="primary"
            icon={<SearchOutlined />}
            loading={loading}
            disabled={fileList.length === 0}
            onClick={handleInspect}
          >
            分析数据
          </Button>
          {loops.length > 1 && (
            <Select
              value={selectedLoop}
              onChange={handleSwitchLoop}
              options={loops.map((l) => ({ label: l.prefix || '(默认)', value: l.prefix }))}
              style={{ width: 200 }}
              placeholder="切换回路"
            />
          )}
          {loops.length > 0 && (
            <Select
              value={loopType || undefined}
              onChange={handleSwitchLoopType}
              options={LOOP_TYPE_OPTIONS}
              style={{ width: 200 }}
              placeholder="选择回路类型"
              suffixIcon={
                loopTypeInferred ? (
                  <Tag color="success" style={{ marginRight: -4 }}>自动</Tag>
                ) : undefined
              }
            />
          )}
        </Space>
      </ProCard>

      {error && (
        <Alert
          type="error"
          message="分析失败"
          description={error}
          style={{ marginBottom: 16 }}
        />
      )}

      {meta && (
        <ProCard style={{ marginBottom: 16 }}>
          <Descriptions column={4}>
            <Descriptions.Item label="数据行数">{meta.rows}</Descriptions.Item>
            <Descriptions.Item label="采样周期">{meta.dt.toFixed(2)} s</Descriptions.Item>
            {meta.steps !== undefined && (
              <Descriptions.Item label="检测到阶跃">{meta.steps}</Descriptions.Item>
            )}
            {meta.usable !== undefined && (
              <Descriptions.Item label="可用窗口">
                {meta.usable} / {windows.length}
              </Descriptions.Item>
            )}
          </Descriptions>
        </ProCard>
      )}

      {csvPath && (
        <ProCard
          title="数据曲线（PV / SV / MV）"
          style={{ marginBottom: 16 }}
          loading={seriesLoading}
        >
          {seriesError && (
            <Alert
              type="error"
              message="曲线加载失败"
              description={seriesError}
              style={{ marginBottom: 12 }}
            />
          )}
          {series && seriesPlotData.length > 0 ? (
            <>
              <Space size="middle" wrap style={{ marginBottom: 12 }}>
                <Tag>总点数：{series.total_points}</Tag>
                <Tag>展示点数：{series.sampled_points}</Tag>
                {series.x_axis === 'timestamp' ? (
                  <DatePicker.RangePicker
                    showTime
                    allowClear
                    value={timeRangeTs}
                    onChange={(v) => setTimeRangeTs((v?.[0] && v?.[1]) ? [v[0], v[1]] : null)}
                  />
                ) : (
                  <div style={{ width: 420 }}>
                    <Slider
                      range
                      min={Number(series.points[0]?.t ?? 0)}
                      max={Number(series.points[series.points.length - 1]?.t ?? 0)}
                      step={Math.max(series.dt, 0.01)}
                      value={timeRangeT ?? undefined}
                      onChange={(v) => {
                        if (Array.isArray(v) && v.length === 2) setTimeRangeT([Number(v[0]), Number(v[1])]);
                      }}
                    />
                  </div>
                )}
              </Space>
              <Line
                data={seriesPlotData}
                xField="t"
                yField="value"
                colorField="series"
                height={320}
                smooth={false}
                legend={{ position: 'top-right' }}
                slider={{}}
                xAxis={{
                  type: series.x_axis === 'timestamp' ? 'timeCat' : 'linear',
                  title: { text: series.x_axis === 'timestamp' ? '时间' : '时间 (s)' },
                }}
                yAxis={{ title: { text: '值' } }}
                color={['#1890ff', '#52c41a', '#faad14']}
              />
            </>
          ) : (
            <Empty
              description="暂无曲线数据"
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          )}
        </ProCard>
      )}

      {loops.length > 0 && (
        <ProCard title="检测到的回路" style={{ marginBottom: 16 }}>
          <Table
            size="small"
            dataSource={loops}
            rowKey="prefix"
            pagination={false}
            columns={[
              {
                title: '回路前缀',
                dataIndex: 'prefix',
                render: (v: string) => <Tag color="blue">{v || '(默认)'}</Tag>,
              },
              { title: 'PV 列', dataIndex: 'pv_col' },
              { title: 'MV 列', dataIndex: 'mv_col' },
              { title: 'SV 列', dataIndex: 'sv_col', render: (v: string) => v || '-' },
            ]}
          />
        </ProCard>
      )}

      {windows.length > 0 && (
        <ProCard title={`候选辨识窗口 (${windows.length})`} style={{ marginBottom: 16 }}>
          <Table<CandidateWindow>
            size="small"
            dataSource={windows}
            rowKey="index"
            pagination={{ pageSize: 8, size: 'small' }}
            onRow={(record) => ({
              onClick: () => setSelectedWindow(record),
              style: { cursor: 'pointer' },
            })}
            rowClassName={(r) =>
              selectedWindow?.index === r.index ? 'ant-table-row-selected' : ''
            }
            columns={[
              { title: '#', dataIndex: 'index', width: 50 },
              {
                title: '可用',
                dataIndex: 'window_usable_for_id',
                width: 80,
                render: (v: boolean) => (
                  <Tag color={v ? 'success' : 'default'}>{v ? '✓' : '×'}</Tag>
                ),
              },
              { title: '起始', dataIndex: 'start', width: 80 },
              { title: '结束', dataIndex: 'end', width: 80 },
              { title: '长度', dataIndex: 'n_points', width: 80 },
              {
                title: '质量分',
                dataIndex: 'score',
                width: 100,
                render: (v: number) => v.toFixed(3),
                sorter: (a, b) => a.score - b.score,
                defaultSortOrder: 'descend' as const,
              },
              {
                title: '幅值',
                dataIndex: 'amplitude',
                width: 100,
                render: (v: number) => v.toFixed(2),
              },
              {
                title: '来源',
                dataIndex: 'source',
                render: (v: string) => <Tag>{v || '-'}</Tag>,
              },
            ]}
          />

          {selectedWindow && previewData.length > 0 ? (
            <div style={{ marginTop: 16 }}>
              <div style={{ color: '#888', fontSize: 12, marginBottom: 8 }}>
                窗口 #{selectedWindow.index} 数据预览（已降采样到 ≤120 点）
              </div>
              <Line
                data={previewData}
                xField="t"
                yField="value"
                colorField="series"
                height={260}
                smooth={false}
                color={['#1890ff', '#faad14']}
                legend={{ position: 'top-right' }}
                xAxis={{ title: { text: '相对时间 (s)' } }}
                yAxis={{ title: { text: '值' } }}
              />
            </div>
          ) : (
            <Empty
              description="点击表格中的窗口查看 PV/MV 预览"
              style={{ marginTop: 24 }}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          )}
        </ProCard>
      )}
    </PageContainer>
  );
}
