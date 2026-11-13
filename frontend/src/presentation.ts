import type { Citation } from './App'

export type PresentationState =
  | 'no_evidence'
  | 'requirement_not_established'
  | 'supported'

export function getPresentationState(
  insufficient?: boolean,
  citations?: Citation[],
): PresentationState {
  if (insufficient) {
    if (citations && citations.length > 0) {
      return 'requirement_not_established'
    }
    return 'no_evidence'
  }
  return 'supported'
}

export function getStatusLabel(state: PresentationState): string | null {
  switch (state) {
    case 'no_evidence':
      return 'Relevant evidence not found in this document.'
    case 'requirement_not_established':
      return 'The document does not specify this requirement in the retrieved passage.'
    case 'supported':
      return null
  }
}
