import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Alert, Button } from 'antd';

interface SectionBoundaryProps {
  label: string;
  children: ReactNode;
}

interface SectionBoundaryState {
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class SectionErrorBoundary extends Component<SectionBoundaryProps, SectionBoundaryState> {
  state: SectionBoundaryState = { error: null, errorInfo: null };

  static getDerivedStateFromError(error: Error): SectionBoundaryState {
    return { error, errorInfo: null };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // eslint-disable-next-line no-console
    console.error(`[SectionErrorBoundary:${this.props.label}]`, error, errorInfo);
    this.setState({ error, errorInfo });
  }

  reset = () => this.setState({ error: null, errorInfo: null });

  render() {
    const { error, errorInfo } = this.state;
    if (error) {
      return (
        <Alert
          type="error"
          showIcon
          message={`${this.props.label} 渲染异常`}
          description={(
            <div style={{ maxHeight: 320, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
              <div style={{ fontWeight: 600 }}>{error.name}: {error.message}</div>
              {error.stack && <pre style={{ fontSize: 12, marginTop: 8 }}>{error.stack}</pre>}
              {errorInfo?.componentStack && (
                <pre style={{ fontSize: 12, marginTop: 8, color: '#888' }}>{errorInfo.componentStack}</pre>
              )}
              <div style={{ marginTop: 8 }}>
                <Button size="small" onClick={this.reset}>清除错误，重新渲染</Button>
              </div>
            </div>
          )}
        />
      );
    }
    return this.props.children as ReactNode;
  }
}
