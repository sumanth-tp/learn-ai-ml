import type {ComponentProps, ReactNode} from 'react';

type IconProps = ComponentProps<'svg'>;

/**
 * Inline, currentColor-driven icons.
 * Kept local (instead of an icon package) so the bundle stays small and both
 * color themes inherit automatically.
 */
function Icon({children, ...props}: IconProps & {children: ReactNode}) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}>
      {children}
    </svg>
  );
}

export function BrainIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 5a2.5 2.5 0 0 0-5 0 2.5 2.5 0 0 0-2 4 2.5 2.5 0 0 0 1 4.5A2.5 2.5 0 0 0 9 19a2.5 2.5 0 0 0 3-1.5Z" />
      <path d="M12 5a2.5 2.5 0 0 1 5 0 2.5 2.5 0 0 1 2 4 2.5 2.5 0 0 1-1 4.5A2.5 2.5 0 0 1 15 19a2.5 2.5 0 0 1-3-1.5Z" />
      <path d="M12 5v12.5" />
    </Icon>
  );
}

export function CodeIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m8 7-5 5 5 5" />
      <path d="m16 7 5 5-5 5" />
      <path d="m13.5 4-3 16" />
    </Icon>
  );
}

export function SigmaIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M18 5H6l6 7-6 7h12" />
      <path d="M18 5V3.5M18 19v1.5" />
    </Icon>
  );
}

export function LayersIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m12 3 9 5-9 5-9-5 9-5Z" />
      <path d="m3 13 9 5 9-5" />
      <path d="m3 17.5 9 5 9-5" />
    </Icon>
  );
}

export function BoltIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M13 2 4.5 13H11l-1 9 8.5-11H12l1-9Z" />
    </Icon>
  );
}

export function ChatIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M21 12a8 8 0 0 1-8 8H8l-5 3 1.2-4.2A8 8 0 1 1 21 12Z" />
      <path d="M9 11h6M9 14.5h3.5" />
    </Icon>
  );
}

export function ChartIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3 3v16.5A1.5 1.5 0 0 0 4.5 21H21" />
      <path d="m7 15 3.5-4.5 3 3L19 7" />
    </Icon>
  );
}

export function BoxIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M20.5 7.5 12 3 3.5 7.5v9L12 21l8.5-4.5v-9Z" />
      <path d="M3.5 7.5 12 12l8.5-4.5M12 12v9" />
    </Icon>
  );
}

export function ArrowIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M5 12h14" />
      <path d="m13 6 6 6-6 6" />
    </Icon>
  );
}

export function SparkIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 3.5 13.7 9l5.5 1.7-5.5 1.8L12 18l-1.7-5.5-5.5-1.8L10.3 9 12 3.5Z" />
    </Icon>
  );
}
