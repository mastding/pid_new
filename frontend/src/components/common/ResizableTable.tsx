import { useMemo, useState } from 'react';
import { Table as AntTable } from 'antd';
import type { TableProps } from 'antd';

type AnyColumn = NonNullable<TableProps<any>['columns']>[number] & {
  children?: AnyColumn[];
  width?: number | string;
};

type HeaderCellProps = React.ThHTMLAttributes<HTMLTableCellElement> & {
  width?: number | string;
  onResize?: (width: number) => void;
};

const MIN_COLUMN_WIDTH = 72;

function columnKey(column: AnyColumn, indexPath: string) {
  const dataIndex = column && 'dataIndex' in column ? column.dataIndex : undefined;
  const title = typeof column.title === 'string' ? column.title : '';
  return String(column.key ?? (Array.isArray(dataIndex) ? dataIndex.join('.') : dataIndex) ?? title ?? indexPath);
}

function ResizableHeaderCell({ width, onResize, children, className, ...rest }: HeaderCellProps) {
  const handlePointerDown = (event: React.PointerEvent<HTMLSpanElement>) => {
    if (!onResize) return;
    event.preventDefault();
    event.stopPropagation();

    const th = event.currentTarget.closest('th');
    const startX = event.clientX;
    const startWidth = typeof width === 'number'
      ? width
      : Math.round(th?.getBoundingClientRect().width || MIN_COLUMN_WIDTH);

    const handlePointerMove = (moveEvent: PointerEvent) => {
      const nextWidth = Math.max(MIN_COLUMN_WIDTH, Math.round(startWidth + moveEvent.clientX - startX));
      onResize(nextWidth);
    };

    const handlePointerUp = () => {
      document.body.classList.remove('is-resizing-table-column');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };

    document.body.classList.add('is-resizing-table-column');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
  };

  return (
    <th {...rest} className={`${className ?? ''} resizable-table-th`} style={{ ...rest.style, position: 'relative' }}>
      {children}
      {onResize && (
        <span
          className="resizable-table-handle"
          role="separator"
          aria-orientation="vertical"
          style={{
            position: 'absolute',
            top: 0,
            right: -4,
            zIndex: 3,
            width: 8,
            height: '100%',
            cursor: 'col-resize',
            userSelect: 'none',
            touchAction: 'none',
          }}
          onPointerDown={handlePointerDown}
        />
      )}
    </th>
  );
}

export function ResizableTable<RecordType extends object = object>(props: TableProps<RecordType>) {
  const { columns, components, scroll, ...rest } = props;
  const [columnWidths, setColumnWidths] = useState<Record<string, number>>({});

  const resizableColumns = useMemo(() => {
    const enhance = (items: TableProps<RecordType>['columns'], parentPath = ''): TableProps<RecordType>['columns'] => (
      items?.map((column, index) => {
        const anyColumn = column as unknown as AnyColumn;
        const key = columnKey(anyColumn, `${parentPath}${index}`);
        const width = columnWidths[key] ?? anyColumn.width;
        const nextColumn: AnyColumn = {
          ...anyColumn,
          width,
          onHeaderCell: (col: unknown) => {
            const base = typeof anyColumn.onHeaderCell === 'function'
              ? anyColumn.onHeaderCell(col as never)
              : {};
            return {
              ...base,
              width,
              onResize: (nextWidth: number) => setColumnWidths((prev) => ({ ...prev, [key]: nextWidth })),
            };
          },
        };
        if (anyColumn.children?.length) {
          nextColumn.children = enhance(anyColumn.children as TableProps<RecordType>['columns'], `${parentPath}${index}.`) as AnyColumn[];
        }
        return nextColumn as unknown as NonNullable<TableProps<RecordType>['columns']>[number];
      })
    );
    return enhance(columns);
  }, [columns, columnWidths]);

  return (
    <AntTable<RecordType>
      {...rest}
      columns={resizableColumns}
      scroll={{ x: 'max-content', ...scroll }}
      components={{
        ...components,
        header: {
          ...components?.header,
          cell: ResizableHeaderCell,
        },
      }}
    />
  );
}
