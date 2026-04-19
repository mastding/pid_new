import { useState, useCallback } from 'react';
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
} from 'antd';
import { UploadOutlined, SearchOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd';
import { Line } from '@ant-design/charts';
import { inspectLoops, inspectWindows } from '@/services/api';
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
  const [windows, setWindows] = useState<CandidateWindow[]>([]);
  const [selectedWindow, setSelectedWindow] = useState<CandidateWindow | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<{
    rows: number;
    dt: number;
    steps?: number;
    usable?: number;
  } | null>(null);

  const handleInspect = useCallback(async () => {
    const file = fileList[0]?.originFileObj as File | undefined;
    if (!file) {
      message.warning('请先上传 CSV 文件');
      return;
    }
    setLoading(true);
    setError(null);
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

      // 自动加载第一个回路的窗口
      const wResp: InspectWindowsResp = await inspectWindows(file, firstPrefix || undefined);
      if (wResp.error) {
        setError(wResp.error);
      } else {
        setWindows(wResp.windows);
        setMeta({
          rows: wResp.total_rows,
          dt: wResp.sampling_time,
          steps: wResp.step_events,
          usable: wResp.usable_count,
        });
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [fileList]);

  const handleSwitchLoop = useCallback(
    async (prefix: string) => {
      const file = fileList[0]?.originFileObj as File | undefined;
      if (!file) return;
      setSelectedLoop(prefix);
      setLoading(true);
      try {
        const wResp: InspectWindowsResp = await inspectWindows(file, prefix || undefined);
        if (wResp.error) setError(wResp.error);
        else {
          setWindows(wResp.windows);
          setSelectedWindow(null);
          setMeta({
            rows: wResp.total_rows,
            dt: wResp.sampling_time,
            steps: wResp.step_events,
            usable: wResp.usable_count,
          });
        }
      } finally {
        setLoading(false);
      }
    },
    [fileList],
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
