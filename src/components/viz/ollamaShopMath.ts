export const SHOP_INVENTORY: Record<string, {stock: number; basePrice: number}> = {
  laptop: {stock: 5, basePrice: 1200},
  monitor: {stock: 0, basePrice: 300},
  keyboard: {stock: 25, basePrice: 80},
};

export function shopTrace(product: string, years: number, continueAfterInventory: boolean) {
  const item = SHOP_INVENTORY[product];
  const discount = Math.min(years * 0.05, 0.30);
  const price = item ? Math.round(item.basePrice * (1 - discount) * 100) / 100 : null;
  const canCalculate = continueAfterInventory && Boolean(item);
  const steps = [
    ['1', 'Model requests inventory', `check_inventory(${product})`],
    ['1', 'Python returns inventory', item ? `stock ${item.stock}; base price ${item.basePrice}` : 'stock 0; base price None'],
  ];
  if (canCalculate) {
    steps.push(
      ['2', 'Model requests discount', `calculate_loyalty_discount(${item.basePrice}, ${years})`],
      ['2', 'Python returns price', String(price)],
      ['3', 'Model writes the answer', `Computed price ${price}; stock ${item.stock}`],
    );
  } else {
    steps.push(['2', 'No executed discount', item ? 'Inventory alone cannot establish the discounted price.' : 'No base price: the discount function lacks an input.']);
  }
  return {item, discountPercent: Math.round(discount * 100), price, computedPrice: canCalculate ? price : null, steps};
}
