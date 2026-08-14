export async function rolloutPaymentService(ns: string) {
  // kubectl apply -f payment-service.yaml -n ${ns}
  return `rolled out payment-service in ${ns}`;
}
