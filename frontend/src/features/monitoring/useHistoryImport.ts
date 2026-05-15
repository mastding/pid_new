import { useState } from 'react';
import { message } from 'antd';
import type { UploadFile } from 'antd';
import { importHistoryFiles } from '@/services/api';

interface UseHistoryImportOptions {
  loadLoops: () => Promise<void>;
  selectLoop: (loopId?: string) => void;
}

export function useHistoryImport({ loadLoops, selectLoop }: UseHistoryImportOptions) {
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [dataSourceType, setDataSourceType] = useState<string>('history_upload');
  const [importing, setImporting] = useState(false);

  const handleImport = async () => {
    const files = fileList.map((item) => item.originFileObj).filter(Boolean) as File[];
    if (!files.length) {
      message.warning('请先选择历史数据文件');
      return;
    }

    setImporting(true);
    try {
      const resp = await importHistoryFiles(files);
      message.success(`导入 ${resp.imported_count} 个回路`);
      if (resp.errors.length) message.warning(`${resp.errors.length} 个文件导入失败，请检查格式`);
      setFileList([]);
      await loadLoops();
      selectLoop(resp.loops[0]?.loop_id);
    } catch (error) {
      message.error(`导入失败：${String(error)}`);
    } finally {
      setImporting(false);
    }
  };

  return {
    dataSourceType,
    fileList,
    importing,
    handleImport,
    setDataSourceType,
    setFileList,
  };
}
