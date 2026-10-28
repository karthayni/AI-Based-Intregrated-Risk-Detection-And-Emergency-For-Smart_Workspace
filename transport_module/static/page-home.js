start(() => {
  const o = overallStatus();
  $('#board').className = 'board mini ' + (o.level === 'normal' ? '' : o.level);
  $('#bTitle').textContent = o.title;
  $('#bText').textContent = App.ppe && !App.ppe.available ? 'The workplace log is not connected yet. ' + (App.state ? 'Transport is being tracked.' : '') : o.text;
  $('#bHour').textContent = App.ppe && App.ppe.available ? App.ppe.last_hour : '-';
  $('#bTrans').textContent = App.state ? o.active : '-';
});
