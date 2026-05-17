import { useCallback, useEffect, useState } from 'react';
import { Form, message } from 'antd';

import {
  fetchModelConfig,
  fetchPolicyConfig,
  fetchPromptConfig,
  resetPromptConfig,
  testModelConfig,
  updateModelConfig,
  updatePromptConfig,
  type ModelConfig,
  type PolicyConfig,
  type PromptConfig,
} from '@/services/api';
import type { PromptConfigField } from './promptConfigItems';

function applyPromptConfigToForm(form: ReturnType<typeof Form.useForm>[0], config: PromptConfig) {
  form.setFieldsValue({
    assistant_system_prompt: config.assistant_system_prompt || '',
    assistant_developer_prompt: config.assistant_developer_prompt || '',
    assistant_response_schema: config.assistant_response_schema || '',
    window_policy_system_prompt: config.window_policy_system_prompt || '',
    window_policy_user_prompt_template: config.window_policy_user_prompt_template || '',
    identification_review_system_prompt: config.identification_review_system_prompt || '',
    identification_review_user_prompt_template: config.identification_review_user_prompt_template || '',
  });
}

export function useSettingsConfigs(activeSub?: string) {
  const [modelConfig, setModelConfig] = useState<ModelConfig | null>(null);
  const [modelConfigLoading, setModelConfigLoading] = useState(false);
  const [modelConfigSaving, setModelConfigSaving] = useState(false);
  const [modelConfigTesting, setModelConfigTesting] = useState(false);
  const [modelConfigForm] = Form.useForm();
  const [modelConfigTestResult, setModelConfigTestResult] = useState<{
    status: string;
    message: string;
  } | null>(null);

  const [policyConfig, setPolicyConfig] = useState<PolicyConfig | null>(null);
  const [policyConfigLoading, setPolicyConfigLoading] = useState(false);

  const [promptConfig, setPromptConfig] = useState<PromptConfig | null>(null);
  const [promptConfigLoading, setPromptConfigLoading] = useState(false);
  const [promptConfigSaving, setPromptConfigSaving] = useState(false);
  const [promptConfigForm] = Form.useForm();
  const [activePromptField, setActivePromptField] = useState<PromptConfigField>('assistant_system_prompt');

  const loadModelConfig = useCallback(async () => {
    setModelConfigLoading(true);
    try {
      const data = await fetchModelConfig();
      setModelConfig(data);
      setModelConfigTestResult(null);
    } catch {
      message.error('加载模型配置失败');
    } finally {
      setModelConfigLoading(false);
    }
  }, []);

  const saveModelConfig = useCallback(async (values: Record<string, unknown>) => {
    setModelConfigSaving(true);
    try {
      const body: Record<string, string | null> = {};
      const prevMaskedKey = modelConfig?.model_api_key || '';
      for (const k of ['model_api_url', 'model_api_key', 'model_name']) {
        const v = String(values[k] ?? '').trim();
        if (k === 'model_api_key' && v === prevMaskedKey) {
          body[k] = null;
        } else {
          body[k] = v || null;
        }
      }
      const resp = await updateModelConfig(body);
      setModelConfig(resp.config);
      modelConfigForm.setFieldsValue({
        model_api_url: resp.config.model_api_url || '',
        model_name: resp.config.model_name || '',
        model_api_key: resp.config.model_api_key || '',
      });
      setModelConfigTestResult(null);
      message.success('模型配置已保存并生效');
    } catch (e) {
      message.error(`保存失败: ${(e as Error).message}`);
    } finally {
      setModelConfigSaving(false);
    }
  }, [modelConfig, modelConfigForm]);

  const testModelConnection = useCallback(async () => {
    setModelConfigTesting(true);
    setModelConfigTestResult(null);
    try {
      const resp = await testModelConfig();
      setModelConfigTestResult(resp);
      if (resp.status === 'ok') {
        message.success('连接测试通过');
      } else {
        message.warning('连接测试失败，请检查配置');
      }
    } catch (e) {
      setModelConfigTestResult({ status: 'error', message: (e as Error).message });
      message.error('连接测试异常');
    } finally {
      setModelConfigTesting(false);
    }
  }, []);

  const loadPolicyConfig = useCallback(async () => {
    setPolicyConfigLoading(true);
    try {
      const data = await fetchPolicyConfig();
      setPolicyConfig(data);
    } catch (error) {
      message.error(`加载规则配置失败：${String(error)}`);
    } finally {
      setPolicyConfigLoading(false);
    }
  }, []);

  const loadPromptConfig = useCallback(async () => {
    setPromptConfigLoading(true);
    try {
      const data = await fetchPromptConfig();
      setPromptConfig(data);
      applyPromptConfigToForm(promptConfigForm, data);
    } catch (error) {
      message.error(`加载提示词配置失败：${String(error)}`);
    } finally {
      setPromptConfigLoading(false);
    }
  }, [promptConfigForm]);

  const savePromptConfig = useCallback(async () => {
    setPromptConfigSaving(true);
    try {
      const values = promptConfigForm.getFieldsValue(true) as Record<string, unknown>;
      const resp = await updatePromptConfig({
        assistant_system_prompt: String(values.assistant_system_prompt ?? '').trim(),
        assistant_developer_prompt: String(values.assistant_developer_prompt ?? '').trim(),
        assistant_response_schema: String(values.assistant_response_schema ?? '').trim(),
        window_policy_system_prompt: String(values.window_policy_system_prompt ?? '').trim(),
        window_policy_user_prompt_template: String(values.window_policy_user_prompt_template ?? '').trim(),
        identification_review_system_prompt: String(values.identification_review_system_prompt ?? '').trim(),
        identification_review_user_prompt_template: String(values.identification_review_user_prompt_template ?? '').trim(),
      });
      setPromptConfig(resp.config);
      applyPromptConfigToForm(promptConfigForm, resp.config);
      message.success('提示词配置已保存');
    } catch (error) {
      message.error(`保存提示词配置失败：${String(error)}`);
    } finally {
      setPromptConfigSaving(false);
    }
  }, [promptConfigForm]);

  const restoreDefaultPromptConfig = useCallback(async () => {
    setPromptConfigSaving(true);
    try {
      const resp = await resetPromptConfig();
      setPromptConfig(resp.config);
      applyPromptConfigToForm(promptConfigForm, resp.config);
      message.success('已恢复默认提示词');
    } catch (error) {
      message.error(`恢复默认提示词失败：${String(error)}`);
    } finally {
      setPromptConfigSaving(false);
    }
  }, [promptConfigForm]);

  const fillModelConfigForm = useCallback(() => {
    if (!modelConfig) return;
    modelConfigForm.setFieldsValue({
      model_api_url: modelConfig.model_api_url || '',
      model_name: modelConfig.model_name || '',
      model_api_key: modelConfig.model_api_key || '',
    });
  }, [modelConfig, modelConfigForm]);

  useEffect(() => {
    if (activeSub === 'model_config' && !modelConfig) {
      loadModelConfig();
    }
  }, [activeSub, modelConfig, loadModelConfig]);

  useEffect(() => {
    if (activeSub === 'rule_config' && !policyConfig) {
      loadPolicyConfig();
    }
  }, [activeSub, policyConfig, loadPolicyConfig]);

  useEffect(() => {
    if (activeSub === 'prompt_config' && !promptConfig) {
      loadPromptConfig();
    }
  }, [activeSub, promptConfig, loadPromptConfig]);

  useEffect(() => {
    if (modelConfig) {
      fillModelConfigForm();
    }
  }, [fillModelConfigForm, modelConfig]);

  return {
    modelConfig,
    modelConfigLoading,
    modelConfigSaving,
    modelConfigTesting,
    modelConfigForm,
    modelConfigTestResult,
    loadModelConfig,
    saveModelConfig,
    testModelConnection,
    fillModelConfigForm,
    policyConfig,
    policyConfigLoading,
    loadPolicyConfig,
    promptConfig,
    promptConfigLoading,
    promptConfigSaving,
    promptConfigForm,
    activePromptField,
    setActivePromptField,
    loadPromptConfig,
    savePromptConfig,
    restoreDefaultPromptConfig,
  };
}
